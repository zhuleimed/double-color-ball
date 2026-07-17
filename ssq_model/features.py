"""增强特征工程模块。

从数据库读取双色球开奖历史数据，构建用于 LSTM-Transformer 模型和
树模型的特征矩阵。

红球特征（时序窗口，每期约 130 维）:
  1. 基础红球 (6维): 标准化到 [0,1] 的 6 个红球号码
  2. 频次统计 (33维): 每个号码 1-33 在最近 N 期的出现次数
  3. 遗漏期数 (33维): 每个号码最近一次出现距今的期数
  4. 区间分布 (3维): 1-11/12-22/23-33 各区出号数
  5. 和值+跨度+AC值 (3维): 号码整体统计特征
  6. 奇偶比+大小比+质合比 (3维): 属性比例
  7. 连号数 (1维): 连续号码的对数
  总和 ≈ 82 维/期 × window 期 = 序列特征

蓝球特征（单期截面，约 50 维）:
  8. 蓝球频次 (16维): 蓝球 1-16 的滚动频次
  9. 蓝球遗漏 (16维): 蓝球 1-16 的遗漏期数
  10. 红球汇总特征 (若干维): 上期红球的统计特征

关键原则: 所有特征仅使用目标期之前的数据，严格避免 look-ahead bias。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ssq_model.config import ModelConfig, get_config
from ssq_sync.engine import DataEngine
from ssq_sync.logger import get_logger

logger = get_logger(__name__)

# ── 质数集合 (1-33 中的质数) ──
_PRIMES_1_33 = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}


# ════════════════════════════════════════════════════════════
#  红球特征工程
# ════════════════════════════════════════════════════════════

def build_red_features(
    df: pd.DataFrame | None = None,
    config: ModelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """构建红球 LSTM-Transformer 模型的训练数据。

    为每个时间步（从 window 到 N-1）生成：
      - 一个长度为 window 的特征序列（每期含多维特征）
      - 6 个红球标签（分类目标，0-32 的整数）

    Args:
        df: 开奖数据 DataFrame（含 red1-6 列），None 则从数据库加载。
        config: 模型配置，None 使用默认。

    Returns:
        (X, y) 元组:
          X: (n_samples, window, n_features) — LSTM 输入
          y: (n_samples, 6) — 6 个红球标签（0-32 的整数，用于分类）
    """
    if config is None:
        config = get_config()
    if df is None:
        engine = DataEngine()
        df = engine.get_all_draws_df()

    n = len(df)
    window = config.window
    logger.info(f"红球特征: 总数据 {n} 期, 窗口 {window}, 预计样本 {n - window}")

    # ── 1. 基础红球数组 ──
    red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
    red_arr = df[red_cols].values.astype(np.float32)  # (n, 6)

    # ── 2. 预计算逐期特征（每期独立，只用历史数据） ──
    # 每个时间步 i 的特征只用到 0..i 的数据
    per_draw_features = _compute_per_draw_features(red_arr, config)

    n_features = per_draw_features.shape[1]  # 约 82 维
    logger.info(f"每期特征维度: {n_features}")

    # ── 3. 构建序列样本 ──
    X_list, y_list = [], []
    for i in range(window, n):
        # 序列特征: 取 [i-window, i-1] 共 window 期的特征
        seq = per_draw_features[i - window:i]  # (window, n_features)
        X_list.append(seq)

        # 标签: 第 i 期的红球（减1转为 0-32 的类别标签）
        labels = np.clip(red_arr[i].astype(int) - 1, 0, 32)
        y_list.append(labels)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    logger.info(f"红球特征完成: X={X.shape}, y={y.shape}")
    return X, y


def _compute_per_draw_features(
    red_arr: np.ndarray, config: ModelConfig
) -> np.ndarray:
    """为每一期计算特征向量（仅使用该期及之前的数据，无 look-ahead）。

    Args:
        red_arr: (n, 6) 红球数组。
        config: 模型配置。

    Returns:
        (n, n_features) 逐期特征矩阵。
    """
    n = len(red_arr)
    fw = config.freq_window  # 频次窗口
    features_list = []

    # 维护滚动统计
    freq = np.zeros(33, dtype=np.float32)      # 号码频次
    last_seen = np.full(33, -1, dtype=np.int32)  # 上次出现位置
    zone1_hist = []  # 区间1历史
    zone2_hist = []
    zone3_hist = []
    sum_hist = []
    span_hist = []
    odd_ratio_hist = []
    big_ratio_hist = []
    prime_ratio_hist = []
    consec_hist = []

    for i in range(n):
        reds = red_arr[i].astype(int)

        # 更新统计
        for r in reds:
            idx = r - 1  # 0-based
            freq[idx] += 1
            last_seen[idx] = i

        # ── 当前期的特征计算 ──
        feats = []

        # 2a. 频次统计 (33维): 每个号码在当前时刻的出现次数（归一化）
        freq_norm = freq / max(freq.max(), 1)
        feats.extend(freq_norm.tolist())

        # 2b. 遗漏期数 (33维): 当前期 - 上次出现位置
        omit = np.where(last_seen >= 0, i - last_seen, i + 1).astype(np.float32)
        omit_norm = omit / max(omit.max(), 1)
        feats.extend(omit_norm.tolist())

        # 2c. 当前期红球标准化的号码值 (6维)
        reds_norm = (reds - 1) / 32.0  # 归一化到 [0,1]
        feats.extend(reds_norm.tolist())

        # 2d. 区间分布 (3维): 基于当前期
        z1 = np.sum((reds >= 1) & (reds <= 11))
        z2 = np.sum((reds >= 12) & (reds <= 22))
        z3 = np.sum((reds >= 23) & (reds <= 33))
        zone1_hist.append(z1)
        zone2_hist.append(z2)
        zone3_hist.append(z3)
        feats.extend([z1 / 6.0, z2 / 6.0, z3 / 6.0])

        # 2e. 和值、跨度、AC值 (3维)
        s = int(np.sum(reds))
        span = int(reds[5] - reds[0])
        ac = _compute_ac(reds)
        sum_hist.append(s)
        span_hist.append(span)
        feats.extend([s / 200.0, span / 33.0, ac / 10.0])

        # 2f. 奇偶比、大小比、质合比 (3维)
        odd = np.sum(reds % 2 == 1)
        big = np.sum(reds >= 17)
        prime = np.sum([1 for r in reds if r in _PRIMES_1_33])
        odd_ratio_hist.append(odd)
        big_ratio_hist.append(big)
        prime_ratio_hist.append(prime)
        feats.extend([odd / 6.0, big / 6.0, prime / 6.0])

        # 2g. 连号数 (1维): 当前期连续号码对数
        consec = int(np.sum(np.diff(sorted(reds)) == 1))
        consec_hist.append(consec)
        feats.append(consec / 5.0)  # 最多5对

        features_list.append(np.array(feats, dtype=np.float32))

    result = np.array(features_list, dtype=np.float32)

    # 添加滚动窗口统计（对某些特征做平滑）
    # 频次滚动均值 (33维) - 对 freq 做 EMA 平滑
    # 这里暂时保留基础版本，后续可在 Optuna 中搜索是否添加

    return result


def _compute_ac(reds: np.ndarray) -> int:
    """计算双色球 AC 值（算术复杂度）。

    AC 值 = 所有两个号码之差的不同值的个数 - (6-1)

    Args:
        reds: 6 个红球号码（已排序）。

    Returns:
        AC 值，范围 0-10。
    """
    diffs = set()
    for i in range(6):
        for j in range(i + 1, 6):
            diffs.add(int(reds[j] - reds[i]))
    return max(0, len(diffs) - 5)


# ════════════════════════════════════════════════════════════
#  蓝球特征工程
# ════════════════════════════════════════════════════════════

def build_blue_features(
    df: pd.DataFrame | None = None,
    config: ModelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """构建蓝球模型的训练数据。

    蓝球模型使用传统的表格特征（非时序），适合树模型（XGBoost/LightGBM等）。

    特征包括：
      - 蓝球历史频次 (16维)
      - 蓝球遗漏期数 (16维)
      - 上期红球统计特征（和值、跨度、奇偶比等）
      - 上期红球区间分布
      - 红球与蓝球历史关联特征

    Args:
        df: 开奖数据 DataFrame，None 则从数据库加载。
        config: 模型配置。

    Returns:
        (X, y) 元组:
          X: (n_samples, n_features) — 树模型输入
          y: (n_samples,) — 蓝球标签（0-15 的整数）
    """
    if config is None:
        config = get_config()
    if df is None:
        engine = DataEngine()
        df = engine.get_all_draws_df()

    n = len(df)
    red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
    red_arr = df[red_cols].values.astype(int)
    blue_arr = df["blue"].values.astype(int)

    # 滚动统计
    blue_freq = np.zeros(16, dtype=np.float32)
    blue_last_seen = np.full(16, -1, dtype=np.int32)

    X_list, y_list = [], []

    for i in range(1, n):  # 从第 1 期开始（需要上一期数据）
        feats = []

        # ── 8. 蓝球频次 (16维) ──
        blue_freq_norm = blue_freq / max(blue_freq.max(), 1)
        feats.extend(blue_freq_norm.tolist())

        # ── 9. 蓝球遗漏 (16维) ──
        blue_omit = np.where(
            blue_last_seen >= 0,
            i - blue_last_seen,
            i + 1
        ).astype(np.float32)
        blue_omit_norm = blue_omit / max(blue_omit.max(), 1)
        feats.extend(blue_omit_norm.tolist())

        # ── 10. 上期红球统计特征 ──
        prev_reds = red_arr[i - 1]
        # 和值
        feats.append(float(np.sum(prev_reds)) / 200.0)
        # 跨度
        feats.append(float(prev_reds[5] - prev_reds[0]) / 33.0)
        # AC值
        feats.append(_compute_ac(prev_reds) / 10.0)
        # 奇偶比
        feats.append(float(np.sum(prev_reds % 2 == 1)) / 6.0)
        # 大小比
        feats.append(float(np.sum(prev_reds >= 17)) / 6.0)
        # 质合比
        prime_cnt = sum(1 for r in prev_reds if r in _PRIMES_1_33)
        feats.append(float(prime_cnt) / 6.0)

        # ── 上期区间分布 (3维) ──
        z1 = float(np.sum((prev_reds >= 1) & (prev_reds <= 11))) / 6.0
        z2 = float(np.sum((prev_reds >= 12) & (prev_reds <= 22))) / 6.0
        z3 = float(np.sum((prev_reds >= 23) & (prev_reds <= 33))) / 6.0
        feats.extend([z1, z2, z3])

        # ── 红蓝关联特征 ──
        # 上期红球各区是否与蓝球有"跟随"关系（简化：记录上期各区的出号强度）
        # 此处可后续扩展更复杂的关联挖掘
        feats.append(z1)  # 上期小号区强度
        feats.append(z2)  # 上期中号区强度
        feats.append(z3)  # 上期大号区强度

        # 蓝球滚动均值/方差
        if i >= 10:
            recent_blues = blue_arr[max(0, i-10):i]
            feats.append(float(np.mean(recent_blues)) / 16.0)
            feats.append(float(np.std(recent_blues)) / 16.0)
        else:
            feats.extend([0.0, 0.0])

        X_list.append(np.array(feats, dtype=np.float32))
        y_list.append(blue_arr[i] - 1)  # 0-15 的标签

        # 更新滚动统计（用当前期的蓝球）
        curr_blue = blue_arr[i - 1]  # 注意：用上一期的来更新
        blue_freq[curr_blue - 1] += 1
        blue_last_seen[curr_blue - 1] = i - 1

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    logger.info(f"蓝球特征完成: X={X.shape}, y={y.shape}")
    return X, y


# ════════════════════════════════════════════════════════════
#  便捷函数：一次性构建全部特征
# ════════════════════════════════════════════════════════════

def build_all_features(
    config: ModelConfig | None = None,
) -> dict:
    """一次性构建红球和蓝球的全部训练特征。

    Args:
        config: 模型配置。

    Returns:
        dict: {
            "red": {"X": ..., "y": ..., "feature_dim": N},
            "blue": {"X": ..., "y": ..., "feature_dim": N},
            "df": DataFrame,  # 原始数据（用于日期追踪等）
        }
    """
    if config is None:
        config = get_config()

    engine = DataEngine()
    df = engine.get_all_draws_df()
    logger.info(f"加载 {len(df)} 期数据，开始构建特征...")

    red_X, red_y = build_red_features(df, config)
    blue_X, blue_y = build_blue_features(df, config)

    # 对齐样本数（红球和蓝球的样本数可能不同，因为蓝球从第1期开始）
    # 红球: n - window 个样本, 蓝球: n - 1 个样本
    # 截取共有的部分
    min_samples = min(len(red_X), len(blue_X))
    red_X = red_X[-min_samples:]
    red_y = red_y[-min_samples:]
    blue_X = blue_X[-min_samples:]
    blue_y = blue_y[-min_samples:]

    result = {
        "red": {
            "X": red_X,
            "y": red_y,
            "feature_dim": red_X.shape[-1],
            "n_features_per_step": red_X.shape[2],
        },
        "blue": {
            "X": blue_X,
            "y": blue_y,
            "feature_dim": blue_X.shape[1],
        },
        "df": df,
        "n_samples": min_samples,
    }

    logger.info(
        f"特征构建完成: {min_samples} 个对齐样本, "
        f"红球特征维度 {red_X.shape}, 蓝球特征维度 {blue_X.shape}"
    )
    return result


# ════════════════════════════════════════════════════════════
#  预测时的特征构建（单期推理）
# ════════════════════════════════════════════════════════════

def build_prediction_features(
    df: pd.DataFrame,
    config: ModelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """为预测构建输入特征（仅最新 window 期 + 最新蓝球特征）。

    Args:
        df: 完整的历史数据 DataFrame（含最新一期）。
        config: 模型配置。

    Returns:
        (red_input, blue_input):
          red_input: (1, window, n_features) — 红球模型输入
          blue_input: (1, n_features) — 蓝球模型输入
    """
    if config is None:
        config = get_config()

    # 红球特征序列：取最后 window 期的 per-draw features
    red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
    red_arr = df[red_cols].values.astype(np.float32)
    per_draw = _compute_per_draw_features(red_arr, config)

    window = config.window
    red_input = per_draw[-window:].reshape(1, window, -1)

    # 蓝球特征：用最新一期数据计算
    blue_arr = df["blue"].values.astype(int)
    n = len(df)

    # 构建蓝球特征向量
    blue_feats = []
    # 蓝球频次
    blue_freq = np.zeros(16, dtype=np.float32)
    blue_last_seen = np.full(16, -1, dtype=np.int32)
    for i in range(max(0, n - 50), n - 1):
        b = blue_arr[i] - 1
        if 0 <= b < 16:
            blue_freq[b] += 1
            blue_last_seen[b] = i

    blue_freq_norm = blue_freq / max(blue_freq.max(), 1)
    blue_feats.extend(blue_freq_norm.tolist())

    blue_omit = np.where(blue_last_seen >= 0, (n - 1) - blue_last_seen, n).astype(np.float32)
    blue_omit_norm = blue_omit / max(blue_omit.max(), 1)
    blue_feats.extend(blue_omit_norm.tolist())

    # 上期红球统计
    prev_reds = red_arr[-1].astype(int)
    blue_feats.append(float(np.sum(prev_reds)) / 200.0)
    blue_feats.append(float(prev_reds[5] - prev_reds[0]) / 33.0)
    blue_feats.append(_compute_ac(prev_reds) / 10.0)
    blue_feats.append(float(np.sum(prev_reds % 2 == 1)) / 6.0)
    blue_feats.append(float(np.sum(prev_reds >= 17)) / 6.0)
    prime_cnt = sum(1 for r in prev_reds if r in _PRIMES_1_33)
    blue_feats.append(float(prime_cnt) / 6.0)

    z1 = float(np.sum((prev_reds >= 1) & (prev_reds <= 11))) / 6.0
    z2 = float(np.sum((prev_reds >= 12) & (prev_reds <= 22))) / 6.0
    z3 = float(np.sum((prev_reds >= 23) & (prev_reds <= 33))) / 6.0
    blue_feats.extend([z1, z2, z3, z1, z2, z3])

    # 蓝球均值/方差
    if n >= 10:
        recent = blue_arr[max(0, n-11):n-1].astype(np.float32)
        blue_feats.append(float(np.mean(recent)) / 16.0)
        blue_feats.append(float(np.std(recent)) / 16.0)
    else:
        blue_feats.extend([0.0, 0.0])

    blue_input = np.array(blue_feats, dtype=np.float32).reshape(1, -1)

    logger.info(
        f"预测特征: red_input={red_input.shape}, blue_input={blue_input.shape}"
    )
    return red_input, blue_input
