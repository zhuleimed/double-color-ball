"""模型参数配置模块。

统一管理红球模型、蓝球模型、特征工程的所有可配置参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """双色球模型全局配置。

    Attributes:
        db_path: SQLite 数据库路径。
        model_dir: 模型文件保存目录。
        window: 时间序列滑动窗口大小（期数）。
        red_classes: 红球分类数（1-33）。
        blue_classes: 蓝球分类数（1-16）。
        red_ball_count: 红球个数。
        test_ratio: 测试集比例。
        random_seed: 随机种子，保证可复现。
    """

    # ── 路径 ──
    db_path: str = "data/ssq_history.db"
    model_dir: str = "data/models"

    # ── 时间窗口 ──
    window: int = 90  # 滑动窗口期数（通过 Optuna 搜索最优值）

    # ── 分类参数 ──
    red_classes: int = 33   # 红球 1-33（注意：标签需要减1，变为 0-32）
    blue_classes: int = 16  # 蓝球 1-16（标签减1，变为 0-15）
    red_ball_count: int = 6

    # ── 训练参数 ──
    test_ratio: float = 0.1  # 测试集比例
    val_ratio: float = 0.1   # 验证集比例（从训练集中划分）
    random_seed: int = 42

    # ── Optuna 搜索参数 ──
    optuna_n_trials: int = 50       # 超参数搜索试验次数（增强版）
    optuna_n_jobs: int = 4          # 并行 trial 数（4×8=32核，在33核以内）
    optuna_timeout: int = 21600     # 搜索超时（秒），6 小时

    # ── 红球模型架构搜索范围 ──
    lstm_units_range: tuple = (32, 256)
    transformer_heads_range: tuple = (2, 8)
    ff_dim_range: tuple = (64, 512)
    dropout_range: tuple = (0.0, 0.5)
    learning_rate_range: tuple = (1e-5, 1e-2)

    # ── 蓝球模型参数 ──
    blue_n_estimators: int = 200

    # ── 特征工程参数 ──
    freq_window: int = 50    # 频次统计的滚动窗口
    omit_window: int = 30    # 遗漏分析的最大回溯窗口
    hot_cold_threshold: float = 0.5  # 冷热号判定阈值（高于均值=热号）

    @property
    def model_dir_path(self) -> Path:
        """模型保存目录的 Path 对象。"""
        p = Path(self.model_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


# 全局单例
_config: ModelConfig | None = None


def get_config() -> ModelConfig:
    """获取全局 ModelConfig 单例。"""
    global _config
    if _config is None:
        _config = ModelConfig()
    return _config
