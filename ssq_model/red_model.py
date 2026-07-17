"""红球 LSTM-Transformer 模型。

架构: LSTM → TransformerBlock × 2 → 6×Dense(33, softmax)
每个红球位置独立输出一个 33 类的分类概率分布。

支持:
  - Optuna 超参数搜索
  - EarlyStopping + ReduceLROnPlateau
  - Beam Search 约束解码（保证递增+不重复）
  - 模型保存/加载
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling1D, Add,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop

from ssq_model.config import ModelConfig, get_config
from ssq_model.features import build_red_features
from ssq_sync.logger import get_logger

logger = get_logger(__name__)

# 禁用 GPU（仅 CPU 模式）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ════════════════════════════════════════════════════════════
#  训练日志回调
# ════════════════════════════════════════════════════════════

class TrainingLogger(tf.keras.callbacks.Callback):
    """详细的训练过程日志记录器。

    在每个 epoch 结束时输出关键指标，支持:
      - 各位置红球准确率追踪
      - 最佳轮次标记 (★)
      - 过拟合检测（训练/验证差距 > 0.5 时告警）
      - 学习率追踪
      - 早停原因记录
    """

    def __init__(self, trial_name: str = "", log_every: int = 5):
        super().__init__()
        self.trial_name = trial_name
        self.log_every = log_every
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def on_train_begin(self, logs=None):
        logger.info(
            f"[{self.trial_name}] 训练开始 | "
            f"样本={self.params.get('samples', '?')} | "
            f"批次={self.params.get('batch_size', '?')} | "
            f"总轮数={self.params.get('epochs', '?')}"
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch += 1

        val_loss = logs.get("val_loss", float("inf"))
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        if epoch % self.log_every == 0 or is_best or epoch == 1:
            # 各位置准确率
            accs = []
            for i in range(6):
                a = logs.get(f"red_pos_{i}_sparse_categorical_accuracy", 0)
                accs.append(a)
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            best = " ★" if is_best else ""
            logger.info(
                f"[{self.trial_name}] Epoch {epoch:3d}/{self.params['epochs']}{best} | "
                f"loss={logs.get('loss',0):.4f} val={val_loss:.4f} | "
                f"acc=[{accs[0]:.3f}/{accs[1]:.3f}/{accs[2]:.3f}/"
                f"{accs[3]:.3f}/{accs[4]:.3f}/{accs[5]:.3f}] | "
                f"lr={lr:.2e} | 最佳轮={self.best_epoch}"
            )

        # 每 10 轮过拟合检测
        if epoch % 10 == 0:
            gap = val_loss - logs.get("loss", 0)
            if gap > 0.5:
                logger.warning(
                    f"[{self.trial_name}] ⚠ 过拟合: "
                    f"train_loss={logs.get('loss',0):.4f} val_loss={val_loss:.4f} gap={gap:.4f}"
                )

    def on_train_end(self, logs=None):
        logger.info(
            f"[{self.trial_name}] 训练结束 | "
            f"最佳轮={self.best_epoch} | 最佳val_loss={self.best_val_loss:.4f}"
        )


# ════════════════════════════════════════════════════════════
#  Transformer Block
# ════════════════════════════════════════════════════════════

class TransformerBlock(tf.keras.layers.Layer):
    """自注意力 Transformer 模块。

    包含 MultiHeadAttention → Add&Norm → FFN → Add&Norm。
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 ff_dim: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


# ════════════════════════════════════════════════════════════
#  模型构建
# ════════════════════════════════════════════════════════════

def create_red_model(
    window: int = 90,
    n_features: int = 82,
    red_classes: int = 33,
    lstm_units: int = 128,
    lstm_units2: int = 64,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_transformers: int = 2,
    dropout_rate: float = 0.1,
    dense_units: int = 128,
    optimizer: str = "adam",
    learning_rate: float = 0.001,
) -> Model:
    """构建 LSTM-Transformer 红球分类模型。

    Args:
        window: 时间序列窗口长度。
        n_features: 每期特征维度。
        red_classes: 分类数（33，对应号码 1-33）。
        lstm_units: 第一层 LSTM 单元数。
        lstm_units2: 第二层 LSTM 单元数。
        num_heads: MultiHeadAttention 头数。
        ff_dim: Transformer FFN 隐藏维度。
        num_transformers: Transformer 层数。
        dropout_rate: Dropout 比率。
        dense_units: 中间 Dense 层单元数。
        optimizer: 优化器名称 "adam"/"nadam"/"rmsprop"。
        learning_rate: 学习率。

    Returns:
        Keras Model，输出 6 个 (batch, 33) 的概率分布。
    """
    inputs = Input(shape=(window, n_features), name="red_sequence")

    # ── LSTM 编码器 ──
    x = LSTM(lstm_units, return_sequences=True, name="lstm_1")(inputs)
    x = Dropout(dropout_rate)(x)

    # ── Transformer 层 ──
    for i in range(num_transformers):
        transformer = TransformerBlock(
            embed_dim=lstm_units,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f"transformer_{i}",
        )
        x = transformer(x)

    # ── 第二层 LSTM ──
    x = LSTM(lstm_units2, return_sequences=False, name="lstm_2")(x)
    x = Dropout(dropout_rate)(x)

    # ── 共享 Dense 层 ──
    x = Dense(dense_units, activation="relu", name="shared_dense")(x)
    x = Dropout(dropout_rate)(x)

    # ── 6 个并行的分类头（每个红球位置独立） ──
    outputs = []
    for i in range(6):
        head = Dense(red_classes, activation="softmax", name=f"red_pos_{i}")(x)
        outputs.append(head)

    model = Model(inputs=inputs, outputs=outputs, name="red_lstm_transformer")

    # 编译（多输出模型需要为每个输出指定 loss 和 metrics）
    opt = _get_optimizer(optimizer, learning_rate)
    model.compile(
        optimizer=opt,
        loss=["sparse_categorical_crossentropy"] * 6,  # 6 个输出各一个 loss
        metrics={f"red_pos_{i}": ["sparse_categorical_accuracy"] for i in range(6)},
    )

    return model


def _get_optimizer(name: str, lr: float):
    """获取优化器实例。"""
    name_lower = name.lower()
    if name_lower == "adam":
        return Adam(learning_rate=lr)
    elif name_lower == "nadam":
        return Nadam(learning_rate=lr)
    elif name_lower == "rmsprop":
        return RMSprop(learning_rate=lr)
    return Adam(learning_rate=lr)


# ════════════════════════════════════════════════════════════
#  Optuna 超参数搜索
# ════════════════════════════════════════════════════════════

def _build_objective(X, y, config: ModelConfig, n_trials: int):
    """构建 Optuna 目标函数（闭包）。"""

    def objective(trial: optuna.Trial) -> float:
        # 更广的搜索空间 — 充分利用7小时夜间窗口
        lstm_units = trial.suggest_int("lstm_units", 32, 320, step=32)
        lstm_units2 = trial.suggest_int("lstm_units2", 16, 160, step=16)
        num_heads = trial.suggest_int("num_heads", 2, 12, step=2)
        ff_dim = trial.suggest_int("ff_dim", 64, 768, step=64)
        num_transformers = trial.suggest_int("num_transformers", 1, 4)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.6)
        dense_units = trial.suggest_int("dense_units", 32, 384, step=32)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 3e-2, log=True)
        optimizer = trial.suggest_categorical("optimizer", ["adam", "nadam", "rmsprop"])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        # TimeSeriesSplit 验证
        tscv = TimeSeriesSplit(n_splits=3)
        val_losses = []

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = [y[train_idx][:, j] for j in range(6)]
            y_val = [y[val_idx][:, j] for j in range(6)]

            model = create_red_model(
                window=config.window,
                n_features=X.shape[2],
                lstm_units=lstm_units,
                lstm_units2=lstm_units2,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_transformers=num_transformers,
                dropout_rate=dropout_rate,
                dense_units=dense_units,
                optimizer=optimizer,
                learning_rate=learning_rate,
            )

            trial_label = f"T{trial.number}"
            train_logger = TrainingLogger(trial_label, log_every=20)

            early_stop = EarlyStopping(
                monitor="val_loss", patience=10,
                restore_best_weights=True, verbose=0,
            )

            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=batch_size,
                callbacks=[early_stop, train_logger],
                verbose=0,
            )

            best_val = min(history.history["val_loss"])
            best_epoch = history.history["val_loss"].index(best_val) + 1
            val_losses.append(best_val)

            logger.info(
                f"[T{trial.number}] fold完成 | "
                f"最佳val_loss={best_val:.4f} @ epoch {best_epoch} | "
                f"总轮数={len(history.history['val_loss'])}"
            )

            # 清理
            del model
            tf.keras.backend.clear_session()

        return np.mean(val_losses)

    return objective


def search_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    config: ModelConfig | None = None,
    n_trials: int | None = None,
    timeout: int | None = None,
) -> dict:
    """Optuna 超参数搜索。

    Args:
        X: 训练特征 (n_samples, window, n_features)。
        y: 标签 (n_samples, 6)。
        config: 模型配置。
        n_trials: 试验次数，None 使用配置默认值。
        timeout: 搜索超时（秒），None 使用配置默认值。

    Returns:
        最佳参数字典。
    """
    if config is None:
        config = get_config()

    n_trials = n_trials or config.optuna_n_trials
    timeout = timeout or config.optuna_timeout

    logger.info(
        f"Optuna 超参数搜索: {n_trials} trials, "
        f"{config.optuna_n_jobs} 并行, 超时 {timeout}s"
    )

    objective_fn = _build_objective(X, y, config, n_trials)

    # 进度回调：每完成一个 trial 输出汇总
    class TrialProgressCallback:
        def __init__(self):
            self.completed = 0
            self.total = n_trials
            self.best_so_far = float("inf")
            self.start_time = datetime.now()

        def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
            self.completed += 1
            if trial.value is not None and trial.value < self.best_so_far:
                self.best_so_far = trial.value
            elapsed = (datetime.now() - self.start_time).total_seconds()
            eta = (elapsed / self.completed) * (self.total - self.completed) if self.completed > 0 else 0
            logger.info(
                f"[Optuna] Trial {self.completed}/{self.total} 完成 | "
                f"当前值={trial.value:.4f} | 全局最佳={self.best_so_far:.4f} | "
                f"耗时={elapsed:.0f}s | 预计剩余={eta:.0f}s | "
                f"已修剪={len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
            )

    progress_cb = TrialProgressCallback()

    study = optuna.create_study(
        direction="minimize",
        study_name=f"red_lstm_transformer_{datetime.now():%Y%m%d_%H%M}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        n_jobs=config.optuna_n_jobs,
        timeout=timeout,
        callbacks=[progress_cb],
        show_progress_bar=True,
    )

    # 汇总
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    logger.info(f"搜索完成: {completed} 完成, {pruned} 修剪, {failed} 失败")

    best = study.best_params
    logger.info(f"最佳参数: {json.dumps(best, indent=2, ensure_ascii=False)}")
    logger.info(f"最佳损失: {study.best_value:.6f}")

    return {
        "params": best,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
    }


# ════════════════════════════════════════════════════════════
#  模型训练
# ════════════════════════════════════════════════════════════

def train_red_model(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    config: ModelConfig | None = None,
    epochs: int = 200,
    batch_size: int = 64,
    validation_split: float = 0.1,
) -> tuple[Model, dict]:
    """用最佳参数训练最终红球模型。

    Args:
        X: 训练特征。
        y: 标签 (n_samples, 6)。
        params: 超参数字典（来自 search_hyperparams 或手动指定）。
        config: 模型配置。
        epochs: 最大训练轮数。
        batch_size: 批次大小。
        validation_split: 验证集比例。

    Returns:
        (trained_model, history_dict)。
    """
    if config is None:
        config = get_config()

    # 分离测试集
    test_size = int(len(X) * config.test_ratio)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train = [y[:-test_size][:, j] for j in range(6)]
    y_test = [y[-test_size:][:, j] for j in range(6)]

    logger.info(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    logger.info(f"模型参数: {json.dumps({k: v for k, v in params.items() if k != 'batch_size'}, indent=2, ensure_ascii=False)}")

    # 构建模型
    model = create_red_model(
        window=config.window,
        n_features=X.shape[2],
        **params,
    )

    # 计算模型参数量
    total_params = model.count_params()
    logger.info(f"模型参数量: {total_params:,}")

    # 回调
    train_logger = TrainingLogger("最终训练", log_every=5)
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=20,
            restore_best_weights=True, verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=8, min_lr=1e-6, verbose=0,
        ),
        train_logger,
    ]

    # 训练
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # 评估
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    # test_loss 是 [total_loss, acc0, acc1, ..., acc5]
    if isinstance(test_loss, list):
        accs = test_loss[1:]  # 各位置准确率
        avg_acc = np.mean(accs)
        logger.info(f"测试集平均准确率: {avg_acc:.4f}")
        for i, acc in enumerate(accs):
            logger.info(f"  位置 {i+1}: {acc:.4f}")
    else:
        logger.info(f"测试损失: {test_loss}")

    return model, {
        "history": history.history,
        "final_test_loss": test_loss,
    }


# ════════════════════════════════════════════════════════════
#  Beam Search 约束解码
# ════════════════════════════════════════════════════════════

def beam_search_decode(
    probs: list[np.ndarray],
    beam_width: int = 5,
) -> list[list[int]]:
    """使用 Beam Search 从 6 个位置的概率分布中解码出最优号码组合。

    约束条件:
      1. 红球必须递增排序（red1 < red2 < ... < red6）
      2. 红球不能重复

    Args:
        probs: 6 个 (33,) 或 (batch, 33) 的概率数组。
        beam_width: Beam 宽度。

    Returns:
        排序后的红球列表，长度为 beam_width（按概率降序），
        每个元素是 6 个红球号码的列表 [1-33]。
    """
    # 确保是 list of (33,) arrays
    prob_list = [np.squeeze(p) for p in probs]
    if len(prob_list) != 6:
        raise ValueError(f"期望 6 个概率数组，得到 {len(prob_list)}")

    # 初始 beam: [(红球列表, log_prob)]
    beams: list[tuple[list[int], float]] = [([], 0.0)]

    for pos in range(6):
        pos_probs = prob_list[pos]  # (33,)
        # 对数概率（避免数值下溢）
        log_probs = np.log(np.clip(pos_probs, 1e-10, 1.0))

        # 对当前位置的概率排序，取 top-k
        candidates = []
        # 取概率最高的前 N 个号码
        top_k = min(beam_width * 3, 33)
        top_indices = np.argsort(log_probs)[-top_k:][::-1]

        for num_idx in top_indices:
            ball = int(num_idx) + 1  # 转为 1-33

            for prev_reds, prev_log_prob in beams:
                # 约束检查: 递增
                if prev_reds and ball <= prev_reds[-1]:
                    continue
                # 约束检查: 不重复
                if ball in prev_reds:
                    continue

                new_reds = prev_reds + [ball]
                new_prob = prev_log_prob + log_probs[num_idx]
                candidates.append((new_reds, new_prob))

        # 按概率排序，保留 top beam_width
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

        if not beams:
            # 回退：直接取每位置最可能号码
            logger.warning("Beam Search 无合法组合，回退到贪心解码")
            return [greedy_decode(prob_list)]

    # 返回排序后的结果
    beams.sort(key=lambda x: x[1], reverse=True)
    return [reds for reds, _ in beams]


def greedy_decode(probs: list[np.ndarray]) -> list[int]:
    """贪心解码：每个位置取概率最高的号码，简单去重+递增处理。"""
    result = []
    for p in probs:
        p_flat = np.squeeze(p)
        # 按概率降序尝试
        for idx in np.argsort(p_flat)[::-1]:
            ball = int(idx) + 1
            if not result or (ball > result[-1] and ball not in result):
                result.append(ball)
                break
        else:
            # 找不到合法号码，取最小可用
            for b in range(1, 34):
                if b not in result and (not result or b > result[-1]):
                    result.append(b)
                    break

    # 去重补齐
    result = sorted(set(result))
    while len(result) < 6:
        for b in range(1, 34):
            if b not in result:
                result.append(b)
                result.sort()
                break
    return result[:6]


# ════════════════════════════════════════════════════════════
#  模型保存/加载
# ════════════════════════════════════════════════════════════

def save_red_model(model: Model, config: ModelConfig | None = None,
                   version: str | None = None) -> str:
    """保存红球模型到磁盘。

    Args:
        model: Keras Model。
        config: 模型配置。
        version: 版本号，None 自动生成。

    Returns:
        保存目录的路径字符串。
    """
    if config is None:
        config = get_config()

    if version is None:
        version = f"v{datetime.now():%Y%m%d_%H%M}"

    save_dir = config.model_dir_path / "red" / version
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存 Keras 模型
    model_path = save_dir / "model.keras"
    model.save(str(model_path))
    logger.info(f"红球模型已保存: {model_path}")

    return str(save_dir)


def load_red_model(version: str | None = None,
                   config: ModelConfig | None = None) -> Model:
    """加载红球模型。

    Args:
        version: 版本号，None 加载最新版本。
        config: 模型配置。

    Returns:
        Keras Model。
    """
    if config is None:
        config = get_config()

    red_dir = config.model_dir_path / "red"
    if not red_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {red_dir}")

    if version:
        model_path = red_dir / version / "model.keras"
    else:
        # 找最新版本（按目录名排序）
        versions = sorted(
            [d for d in red_dir.iterdir() if d.is_dir()],
            reverse=True,
        )
        if not versions:
            raise FileNotFoundError(f"没有找到红球模型: {red_dir}")
        model_path = versions[0] / "model.keras"

    logger.info(f"加载红球模型: {model_path}")
    return tf.keras.models.load_model(
        str(model_path),
        custom_objects={"TransformerBlock": TransformerBlock},
    )
