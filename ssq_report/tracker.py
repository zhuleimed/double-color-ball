"""累积成功率追踪模块。

维护 JSON 文件追踪预测历史，支持:
  - 红球 0-6 命中分布统计
  - 蓝球命中率
  - 中奖等级分布
  - 滚动窗口统计（近 20/50/100 期）
  - 最佳/最差记录

数据文件: output/success_tracker.json
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from ssq_sync.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_TRACKER_PATH = Path(__file__).resolve().parent.parent / "output" / "success_tracker.json"


class SuccessTracker:
    """预测成功率追踪器。

    用法:
        tracker = SuccessTracker()
        tracker.update(period, red_hits, blue_hit, prize_level)
        stats = tracker.get_stats()
    """

    def __init__(self, tracker_path: str | Path = _DEFAULT_TRACKER_PATH):
        self.tracker_path = Path(tracker_path)
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    # ── 读写 ──

    def _load(self) -> dict:
        """加载追踪数据，文件不存在则初始化。"""
        if self.tracker_path.exists():
            try:
                with open(self.tracker_path, encoding="utf-8") as f:
                    data = json.load(f)
                logger.debug(f"加载追踪数据: {data.get('total_predictions', 0)} 条记录")
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return self._init_data()

    def _init_data(self) -> dict:
        """初始化空白追踪数据。"""
        return {
            "total_predictions": 0,
            "total_draws_compared": 0,
            "history": [],          # 每期对比记录
            "cumulative_stats": {
                "red_0": 0, "red_1": 0, "red_2": 0,
                "red_3": 0, "red_4": 0, "red_5": 0, "red_6": 0,
                "blue_hit": 0, "blue_miss": 0,
                "any_prize": 0,     # 任意中奖
                "prize_6plus1": 0,  # 一等奖
                "prize_6plus0": 0,  # 二等奖
                "prize_5plus1": 0,  # 三等奖
            },
            "best_record": None,    # {"period": ..., "reds": 0, "blue": 0, "prize": "..."}
            "rolling_stats": {
                "last_20": {},
                "last_50": {},
                "last_100": {},
            },
            "updated_at": datetime.now().isoformat(),
        }

    def _save(self) -> None:
        """原子保存。"""
        self._data["updated_at"] = datetime.now().isoformat()
        fd, tmp = tempfile.mkstemp(
            suffix=".json", prefix="tracker_", dir=self.tracker_path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.tracker_path)
        except Exception:
            os.unlink(tmp)
            raise

    # ── 更新 ──

    def update(
        self,
        period: str,
        red_hits: int,
        blue_hit: int,
        prize_level: str,
        pred_reds: list[int] | None = None,
        actual_reds: list[int] | None = None,
        pred_blue: int | None = None,
        actual_blue: int | None = None,
    ) -> None:
        """记录一期对比结果。

        Args:
            period: 期号。
            red_hits: 红球命中数 0-6。
            blue_hit: 蓝球命中 0/1。
            prize_level: 中奖等级描述。
            pred_reds: 预测红球（可选）。
            actual_reds: 实际红球（可选）。
            pred_blue: 预测蓝球（可选）。
            actual_blue: 实际蓝球（可选）。
        """
        # 累计统计
        stats = self._data["cumulative_stats"]
        stats[f"red_{red_hits}"] += 1
        if blue_hit:
            stats["blue_hit"] += 1
        else:
            stats["blue_miss"] += 1

        # 中奖判定
        is_prize = not prize_level.startswith("未中奖")
        if is_prize:
            stats["any_prize"] += 1
        if "一等奖" in prize_level:
            stats["prize_6plus1"] += 1
        elif "二等奖" in prize_level:
            stats["prize_6plus0"] += 1
        elif "三等奖" in prize_level:
            stats["prize_5plus1"] += 1

        # 历史记录
        entry = {
            "period": period,
            "red_hits": red_hits,
            "blue_hit": blue_hit,
            "prize_level": prize_level,
            "date": datetime.now().strftime("%Y-%m-%d"),
        }
        if pred_reds:
            entry["pred_reds"] = pred_reds
        if actual_reds:
            entry["actual_reds"] = actual_reds
        if pred_blue:
            entry["pred_blue"] = pred_blue
        if actual_blue:
            entry["actual_blue"] = actual_blue

        self._data["history"].append(entry)
        self._data["total_predictions"] += 1
        self._data["total_draws_compared"] += 1

        # 最佳记录
        best = self._data["best_record"]
        if best is None or red_hits > best.get("reds", -1) or (
            red_hits == best.get("reds", -1) and blue_hit > best.get("blue", -1)
        ):
            self._data["best_record"] = {
                "period": period,
                "reds": red_hits,
                "blue": blue_hit,
                "prize": prize_level,
            }

        # 更新滚动统计
        self._update_rolling()

        self._save()
        logger.info(
            f"追踪更新: 期号={period}, 红球{red_hits}/6, "
            f"蓝球{'中' if blue_hit else '不中'}, 累计{self._data['total_predictions']}期"
        )

    def _update_rolling(self) -> None:
        """更新滚动窗口统计。"""
        history = self._data["history"]
        for window_name, window_size in [("last_20", 20), ("last_50", 50), ("last_100", 100)]:
            recent = history[-window_size:]
            if not recent:
                self._data["rolling_stats"][window_name] = {}
                continue

            n = len(recent)
            red_dist = {}
            for r in range(7):
                red_dist[f"red_{r}"] = sum(1 for h in recent if h["red_hits"] == r)
                red_dist[f"red_{r}_pct"] = red_dist[f"red_{r}"] / n

            blue_hits = sum(1 for h in recent if h["blue_hit"])
            red_3plus = sum(1 for h in recent if h["red_hits"] >= 3)

            self._data["rolling_stats"][window_name] = {
                "n": n,
                "red_3plus": red_3plus,
                "red_3plus_pct": red_3plus / n,
                "blue_hit": blue_hits,
                "blue_hit_pct": blue_hits / n,
                "red_distribution": red_dist,
            }

    # ── 查询 ──

    def get_stats(self) -> dict:
        """获取当前全部统计。

        Returns:
            dict: 包含累积统计、滚动统计、最佳记录等。
        """
        stats = self._data["cumulative_stats"]
        total = self._data["total_predictions"]

        result = {
            "total_predictions": total,
            "updated_at": self._data["updated_at"],
            "best_record": self._data["best_record"],
        }

        if total > 0:
            result["red_hit_rates"] = {
                f"red_{r}": stats[f"red_{r}"] / total
                for r in range(7)
            }
            result["red_3plus_rate"] = sum(
                stats[f"red_{r}"] for r in range(3, 7)
            ) / total
            result["blue_hit_rate"] = stats["blue_hit"] / total
            result["any_prize_rate"] = stats["any_prize"] / total

        result["rolling"] = {}
        for wname in ["last_20", "last_50", "last_100"]:
            ws = self._data["rolling_stats"].get(wname)
            if ws:
                result["rolling"][wname] = {
                    "n": ws["n"],
                    "red_3plus_pct": ws["red_3plus_pct"],
                    "blue_hit_pct": ws["blue_hit_pct"],
                }

        return result

    def get_history_df(self):
        """获取历史记录的 DataFrame（方便分析）。"""
        import pandas as pd
        return pd.DataFrame(self._data["history"])

    def rebuild_from_db(self, db_path: str | None = None) -> int:
        """从 result_compare 表重建追踪数据（数据迁移或修复时使用）。

        Returns:
            重建的记录数。
        """
        import sqlite3
        if db_path is None:
            from ssq_sync.config import get_settings
            db_path = get_settings().db_path

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT period, pred_red_hits, pred_blue_hit, prize_level "
            "FROM result_compare ORDER BY period"
        ).fetchall()
        conn.close()

        # 重置
        self._data = self._init_data()

        for period, red_hits, blue_hit, prize_level in rows:
            self._data["total_predictions"] += 1
            self._data["total_draws_compared"] += 1
            stats = self._data["cumulative_stats"]
            stats[f"red_{red_hits}"] += 1
            if blue_hit:
                stats["blue_hit"] += 1
            else:
                stats["blue_miss"] += 1
            if not prize_level or not prize_level.startswith("未中奖"):
                stats["any_prize"] += 1
            if prize_level and "一等奖" in prize_level:
                stats["prize_6plus1"] += 1

            self._data["history"].append({
                "period": period,
                "red_hits": red_hits,
                "blue_hit": blue_hit,
                "prize_level": prize_level or "",
                "date": "",
            })

        self._update_rolling()
        self._save()
        logger.info(f"从数据库重建追踪: {len(rows)} 条记录")
        return len(rows)
