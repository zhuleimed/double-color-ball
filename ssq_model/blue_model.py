"""蓝球 Stacking 集成模型。

架构: LightGBM + XGBoost + CatBoost + RandomForest → Stacking(LogisticRegression)
目标: 16 类分类（预测蓝球号码 1-16）

支持:
  - Optuna 超参数搜索（各基模型独立搜索）
  - Stacking 集成训练
  - 模型保存/加载（joblib + JSON）
"""

from __future__ import annotations

import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from ssq_model.config import ModelConfig, get_config
from ssq_model.features import build_blue_features
from ssq_sync.logger import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════
#  模型构建
# ════════════════════════════════════════════════════════════

def _create_lgb(params: dict | None = None) -> object:
    """创建 LightGBM 分类器。"""
    from lightgbm import LGBMClassifier
    defaults = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4,
    }
    if params:
        defaults.update(params)
    return LGBMClassifier(**defaults)


def _create_xgb(params: dict | None = None) -> object:
    """创建 XGBoost 分类器。"""
    from xgboost import XGBClassifier
    defaults = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": 4,
    }
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)


def _create_cat(params: dict | None = None) -> object:
    """创建 CatBoost 分类器。"""
    from catboost import CatBoostClassifier
    defaults = {
        "iterations": 200,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": 0,
        "thread_count": 4,
    }
    if params:
        defaults.update(params)
    return CatBoostClassifier(**defaults)


def _create_rf(params: dict | None = None) -> object:
    """创建 RandomForest 分类器。"""
    defaults = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": 4,
    }
    if params:
        defaults.update(params)
    return RandomForestClassifier(**defaults)


# ════════════════════════════════════════════════════════════
#  Stacking 集成
# ════════════════════════════════════════════════════════════

class BlueStackingModel:
    """蓝球 Stacking 集成模型。

    使用 4 个基模型 + LogisticRegression 元模型进行 Stacking。

    用法:
        model = BlueStackingModel()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        pred = model.predict(X_test)
        model.save("data/models/blue")
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or get_config()
        self.base_models: dict[str, object] = {}
        self.meta_model: LogisticRegression | None = None
        self._fitted = False
        # 标签编码器（冗余，y 已经是 0-15）
        self.label_encoder = LabelEncoder()

    def fit(self, X: np.ndarray, y: np.ndarray,
            base_params: dict | None = None) -> "BlueStackingModel":
        """训练 Stacking 集成模型。

        使用 5 折交叉验证生成基模型的 out-of-fold 预测作为元模型特征。

        Args:
            X: (n_samples, n_features) 特征矩阵。
            y: (n_samples,) 标签（0-15 的整数）。
            base_params: 各基模型的参数字典，key 为模型名。

        Returns:
            self
        """
        n_samples, n_classes = len(X), self.config.blue_classes
        n_models = 4

        # 初始化基模型
        bp = base_params or {}
        self.base_models = {
            "lgb": _create_lgb(bp.get("lgb")),
            "xgb": _create_xgb(bp.get("xgb")),
            "cat": _create_cat(bp.get("cat")),
            "rf": _create_rf(bp.get("rf")),
        }

        # ── 5 折交叉验证生成 oof 特征 ──
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros((n_samples, n_models * n_classes), dtype=np.float32)

        for model_idx, (name, model) in enumerate(self.base_models.items()):
            t0 = time.time()
            logger.info(f"[蓝球] 训练基模型 {model_idx+1}/4: {name} (5折交叉验证)...")
            col_start = model_idx * n_classes
            col_end = col_start + n_classes

            fold_accs = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                model.fit(X_tr, y_tr)
                oof_preds[val_idx, col_start:col_end] = model.predict_proba(X_val)

                # 每折准确率
                fold_pred = np.argmax(model.predict_proba(X_val), axis=1)
                fold_acc = np.mean(fold_pred == y_val)
                fold_accs.append(fold_acc)

            # 用全部数据重新训练（用于最终预测）
            model.fit(X, y)

            elapsed = time.time() - t0
            logger.info(
                f"[蓝球] {name} 完成 | 5折准确率: "
                f"{' / '.join(f'{a:.3f}' for a in fold_accs)} | "
                f"均值={np.mean(fold_accs):.3f} | 耗时={elapsed:.0f}s"
            )

        # ── 训练元模型 ──
        logger.info("[蓝球] 训练元模型 (LogisticRegression)...")
        t0 = time.time()
        self.meta_model = LogisticRegression(
            multi_class="multinomial",
            max_iter=1000,
            random_state=42,
            n_jobs=4,
        )
        self.meta_model.fit(oof_preds, y)

        # 元模型在 oof 上的准确率
        meta_pred = self.meta_model.predict(oof_preds)
        meta_acc = np.mean(meta_pred == y)
        logger.info(
            f"[蓝球] 元模型完成 | oof准确率={meta_acc:.4f} | "
            f"耗时={time.time()-t0:.0f}s | 总样本={n_samples}"
        )

        self._fitted = True
        logger.info(f"[蓝球] Stacking 集成训练全部完成")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率分布。

        Args:
            X: (n_samples, n_features) 特征矩阵。

        Returns:
            (n_samples, 16) 概率矩阵。
        """
        if not self._fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        n_samples = len(X)
        n_classes = self.config.blue_classes
        n_models = len(self.base_models)

        # 基模型预测
        base_probs = np.zeros((n_samples, n_models * n_classes), dtype=np.float32)
        for i, (name, model) in enumerate(self.base_models.items()):
            col_start = i * n_classes
            col_end = col_start + n_classes
            base_probs[:, col_start:col_end] = model.predict_proba(X)

        # 元模型预测
        return self.meta_model.predict_proba(base_probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测蓝球类别（0-15）。"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """评估模型性能。

        Returns:
            dict: {"accuracy": ..., "top3_accuracy": ..., ...}
        """
        probs = self.predict_proba(X)
        pred = np.argmax(probs, axis=1)
        accuracy = np.mean(pred == y)

        # Top-3 准确率（预测的前3个概率最高的蓝球中包含实际蓝球）
        top3_preds = np.argsort(probs, axis=1)[:, -3:]
        top3_hit = np.array([y[i] in top3_preds[i] for i in range(len(y))])
        top3_accuracy = np.mean(top3_hit)

        logger.info(f"蓝球准确率: {accuracy:.4f}, Top-3准确率: {top3_accuracy:.4f}")
        return {
            "accuracy": float(accuracy),
            "top3_accuracy": float(top3_accuracy),
        }

    # ── 模型持久化 ──

    def save(self, save_dir: str | Path,
             version: str | None = None) -> str:
        """保存模型到磁盘。

        Args:
            save_dir: 保存目录。
            version: 版本号。

        Returns:
            保存路径字符串。
        """
        save_dir = Path(save_dir)
        if version is None:
            version = f"v{datetime.now():%Y%m%d_%H%M}"
        save_path = save_dir / version
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存基模型（joblib）
        import joblib
        for name, model in self.base_models.items():
            model_path = save_path / f"{name}.pkl"
            joblib.dump(model, str(model_path))

        # 保存元模型
        if self.meta_model:
            joblib.dump(self.meta_model, str(save_path / "meta.pkl"))

        logger.info(f"蓝球模型已保存: {save_path}")
        return str(save_path)

    @classmethod
    def load(cls, load_path: str | Path,
             config: ModelConfig | None = None) -> "BlueStackingModel":
        """加载模型。

        Args:
            load_path: 模型目录路径。
            config: 模型配置。

        Returns:
            BlueStackingModel 实例。
        """
        import joblib
        load_path = Path(load_path)

        instance = cls(config=config)
        instance.base_models = {}

        # 加载基模型
        for name in ["lgb", "xgb", "cat", "rf"]:
            model_path = load_path / f"{name}.pkl"
            if model_path.exists():
                instance.base_models[name] = joblib.load(str(model_path))

        # 加载元模型
        meta_path = load_path / "meta.pkl"
        if meta_path.exists():
            instance.meta_model = joblib.load(str(meta_path))

        instance._fitted = len(instance.base_models) > 0
        logger.info(f"蓝球模型已加载: {load_path}")
        return instance


# ════════════════════════════════════════════════════════════
#  Optuna 超参数搜索
# ════════════════════════════════════════════════════════════

def search_blue_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    config: ModelConfig | None = None,
    n_trials: int = 30,
    timeout: int = 3600,
) -> dict:
    """Optuna 超参数搜索（针对 XGBoost + LightGBM）。

    由于 Stacking 的元模型参数空间太大，这里分别搜索基模型的最优参数。

    Args:
        X: 训练特征 (n_samples, n_features)。
        y: 标签 (n_samples,) 0-15。
        config: 模型配置。
        n_trials: 每模型试验次数。
        timeout: 搜索超时。

    Returns:
        各基模型最佳参数字典。
    """
    if config is None:
        config = get_config()

    best_params = {}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # ── LightGBM 搜索 ──
    logger.info(f"[蓝球Optuna] 搜索 LightGBM ({n_trials} trials)...")
    t0 = time.time()

    def lgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        model = _create_lgb(params)
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(lgb_objective, n_trials=n_trials, timeout=timeout // 2,
                   show_progress_bar=True)
    best_params["lgb"] = study.best_params
    logger.info(f"[蓝球Optuna] LightGBM 完成 | 最佳acc={study.best_value:.4f} | "
                f"耗时={time.time()-t0:.0f}s | 最佳参数: n_estimators={study.best_params.get('n_estimators','?')}, "
                f"lr={study.best_params.get('learning_rate','?'):.4f}, "
                f"leaves={study.best_params.get('num_leaves','?')}")

    # ── XGBoost 搜索 ──
    logger.info(f"[蓝球Optuna] 搜索 XGBoost ({n_trials} trials)...")
    t0 = time.time()

    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = _create_xgb(params)
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(xgb_objective, n_trials=n_trials, timeout=timeout // 2,
                   show_progress_bar=True)
    best_params["xgb"] = study.best_params
    logger.info(f"[蓝球Optuna] XGBoost 完成 | 最佳acc={study.best_value:.4f} | "
                f"耗时={time.time()-t0:.0f}s | 最佳参数: n_estimators={study.best_params.get('n_estimators','?')}, "
                f"lr={study.best_params.get('learning_rate','?'):.4f}, "
                f"depth={study.best_params.get('max_depth','?')}")

    return best_params


def train_blue_model(
    X: np.ndarray,
    y: np.ndarray,
    base_params: dict | None = None,
    config: ModelConfig | None = None,
) -> tuple[BlueStackingModel, dict]:
    """训练最终蓝球模型。

    Args:
        X: 训练特征。
        y: 标签 0-15。
        base_params: 基模型参数字典。
        config: 模型配置。

    Returns:
        (trained_model, metrics_dict)。
    """
    if config is None:
        config = get_config()

    # 划分测试集
    test_size = int(len(X) * config.test_ratio)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    logger.info(f"蓝球训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 训练
    model = BlueStackingModel(config)
    model.fit(X_train, y_train, base_params)

    # 评估
    metrics = model.evaluate(X_test, y_test)

    return model, metrics
