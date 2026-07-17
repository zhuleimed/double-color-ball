"""多模型集成预测模块。

整合红球 LSTM-Transformer 和蓝球 Stacking 模型，
提供 Beam Search 约束解码和最终号码输出。

核心流程:
  1. 红球模型预测 6×33 概率分布
  2. Beam Search 约束解码 → 最优红球组合
  3. 蓝球模型预测 16 类概率 → 最优蓝球
  4. 组装最终彩票号码
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ssq_model.config import ModelConfig, get_config
from ssq_model.features import build_prediction_features
from ssq_model.red_model import (
    beam_search_decode,
    greedy_decode,
    load_red_model,
)
from ssq_model.blue_model import BlueStackingModel
from ssq_sync.engine import DataEngine
from ssq_sync.logger import get_logger

logger = get_logger(__name__)


class SSQEnsemble:
    """双色球预测集成器。

    包装红球模型和蓝球模型，提供统一的预测接口。

    用法:
        ensemble = SSQEnsemble()
        ensemble.load_models()                    # 加载已训练的模型
        ticket = ensemble.predict_next()          # 预测下一期号码
        # ticket = {"period": "2025084", "reds": [2,8,14,21,28,33], "blue": 15,
        #           "red_confidence": [...], "blue_confidence": 0.73}
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or get_config()
        self.red_model = None       # Keras Model
        self.blue_model = None      # BlueStackingModel
        self._loaded = False

    # ── 模型加载 ──

    def load_models(self, red_version: str | None = None,
                    blue_version: str | None = None) -> None:
        """加载红球和蓝球模型。

        Args:
            red_version: 红球模型版本号，None=最新。
            blue_version: 蓝球模型版本号，None=最新。
        """
        logger.info("加载模型...")

        # 红球模型
        try:
            self.red_model = load_red_model(red_version, self.config)
            logger.info("红球模型已加载")
        except FileNotFoundError as e:
            logger.warning(f"红球模型未找到: {e}")
            self.red_model = None

        # 蓝球模型
        try:
            blue_dir = self.config.model_dir_path / "blue"
            if blue_version:
                blue_path = blue_dir / blue_version
            else:
                versions = sorted(
                    [d for d in blue_dir.iterdir() if d.is_dir()],
                    reverse=True,
                )
                if not versions:
                    raise FileNotFoundError(f"没有找到蓝球模型: {blue_dir}")
                blue_path = versions[0]

            self.blue_model = BlueStackingModel.load(str(blue_path), self.config)
            logger.info("蓝球模型已加载")
        except FileNotFoundError as e:
            logger.warning(f"蓝球模型未找到: {e}")
            self.blue_model = None

        self._loaded = any([self.red_model is not None, self.blue_model is not None])
        if self._loaded:
            logger.info("模型加载完成")
        else:
            logger.error("没有可用的模型！请先运行 train.py 训练模型")

    # ── 预测 ──

    def predict_next(
        self,
        beam_width: int = 5,
        df: pd.DataFrame | None = None,
    ) -> dict:
        """预测下一期双色球号码。

        Args:
            beam_width: Beam Search 宽度（越大越精确但越慢）。
            df: 历史数据 DataFrame，None 从数据库加载。

        Returns:
            dict: {
                "period": 预测期号,
                "reds": [6个红球，1-33，递增排序],
                "blue": 蓝球号码 1-16,
                "red_confidence": [6个置信度],
                "blue_confidence": 蓝球置信度,
                "red_candidates": [备选红球组合列表],
            }
        """
        if not self._loaded:
            raise RuntimeError("模型未加载，请先调用 load_models()")

        # 加载数据
        if df is None:
            engine = DataEngine()
            df = engine.get_all_draws_df()

        # 推断下期期号
        latest_period = str(df["period"].iloc[-1])
        next_period = self._next_period(latest_period)

        # 构建预测特征
        red_input, blue_input = build_prediction_features(df, self.config)

        # ── 红球预测 ──
        red_result = self._predict_reds(red_input, beam_width)

        # ── 蓝球预测 ──
        blue_result = self._predict_blue(blue_input)

        ticket = {
            "period": next_period,
            "reds": red_result["reds"],
            "blue": blue_result["blue"],
            "red_confidence": red_result["confidence"],
            "blue_confidence": blue_result["confidence"],
            "red_candidates": red_result.get("candidates", []),
            "blue_top3": blue_result.get("top3", []),
        }

        logger.info(
            f"预测完成: 期号={next_period}, "
            f"红球={ticket['reds']}, 蓝球={ticket['blue']:02d}"
        )
        return ticket

    def _predict_reds(self, red_input: np.ndarray,
                      beam_width: int) -> dict:
        """红球预测 + Beam Search 解码。

        Args:
            red_input: (1, window, n_features) 输入序列。
            beam_width: Beam Search 宽度。

        Returns:
            dict: {"reds": [...], "confidence": [...], "candidates": [...]}
        """
        if self.red_model is None:
            logger.warning("红球模型不可用，使用随机基线")
            rng = np.random.default_rng()
            reds = sorted(rng.choice(33, size=6, replace=False) + 1)
            return {
                "reds": reds,
                "confidence": [0.03] * 6,
                "candidates": [reds],
            }

        # 模型预测 6×33 概率
        probs = self.red_model.predict(red_input, verbose=0)
        # probs 是 list of (batch, 33) arrays

        # Beam Search 解码
        candidates = beam_search_decode(probs, beam_width=beam_width)

        if not candidates:
            # 回退到贪心解码
            candidates = [greedy_decode(probs)]

        best_reds = candidates[0]

        # 计算每个位置的置信度
        confidence = []
        for i, p in enumerate(probs):
            prob_arr = np.squeeze(p)
            ball_idx = best_reds[i] - 1  # 转 0-based
            confidence.append(float(prob_arr[ball_idx]))

        return {
            "reds": best_reds,
            "confidence": confidence,
            "candidates": candidates[:3],  # 保留前3个候选
        }

    def _predict_blue(self, blue_input: np.ndarray) -> dict:
        """蓝球预测。

        Args:
            blue_input: (1, n_features) 特征向量。

        Returns:
            dict: {"blue": int, "confidence": float, "top3": [...]}
        """
        if self.blue_model is None:
            logger.warning("蓝球模型不可用，使用随机基线")
            rng = np.random.default_rng()
            return {
                "blue": int(rng.integers(1, 17)),
                "confidence": 0.0625,
                "top3": [int(x) for x in rng.choice(16, size=3, replace=False) + 1],
            }

        probs = self.blue_model.predict_proba(blue_input)[0]  # (16,)
        top_indices = np.argsort(probs)[-3:][::-1]

        return {
            "blue": int(top_indices[0]) + 1,
            "confidence": float(probs[top_indices[0]]),
            "top3": [int(i) + 1 for i in top_indices],
        }

    # ── 工具方法 ──

    @staticmethod
    def _next_period(current_period: str) -> str:
        """推算下一期期号。

        Args:
            current_period: 当前最新期号，如 "2025083"。

        Returns:
            下一期期号，如 "2025084"。
        """
        year = int(current_period[:4])
        num = int(current_period[4:]) + 1

        # 每年最多 156 期左右，超过则进位
        if num > 160:
            year += 1
            num = 1

        return f"{year}{num:03d}"

    def save_prediction(self, ticket: dict) -> int:
        """将预测结果保存到数据库 prediction_log 表。

        Args:
            ticket: predict_next() 返回的预测字典。

        Returns:
            插入记录的 ID。
        """
        import sqlite3
        from datetime import date

        conn = sqlite3.connect(str(self.config.db_path))
        try:
            reds = ticket["reds"]
            cur = conn.execute(
                """INSERT INTO prediction_log
                   (period, pred_date, red1, red2, red3, red4, red5, red6, blue,
                    model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticket["period"],
                    date.today().isoformat(),
                    reds[0], reds[1], reds[2], reds[3], reds[4], reds[5],
                    ticket["blue"],
                    f"red_lstm_transformer+blue_stacking",
                ),
            )
            conn.commit()
            row_id = cur.lastrowid
            logger.info(f"预测已保存: ID={row_id}, 期号={ticket['period']}")
            return row_id
        finally:
            conn.close()
