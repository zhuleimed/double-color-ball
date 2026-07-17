"""结果对比模块：预测 vs 实际开奖号码对比。

功能:
  1. 从数据库取出预测和对应实际开奖结果
  2. 计算红球命中数、蓝球命中、具体命中号码
  3. 判定中奖等级
  4. 写入 result_compare 表
"""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import Optional

from ssq_sync.engine import DataEngine
from ssq_sync.logger import get_logger

logger = get_logger(__name__)

# ── 双色球中奖等级判定规则 ──
# (红球命中数, 蓝球命中) → 等级名称
PRIZE_RULES = {
    (6, 1): "一等奖 6+1 🏆",
    (6, 0): "二等奖 6+0",
    (5, 1): "三等奖 5+1 (3000元)",
    (5, 0): "四等奖 5+0 (200元)",
    (4, 1): "四等奖 4+1 (200元)",
    (4, 0): "五等奖 4+0 (10元)",
    (3, 1): "五等奖 3+1 (10元)",
    (2, 1): "六等奖 2+1 (5元)",
    (1, 1): "六等奖 1+1 (5元)",
    (0, 1): "六等奖 0+1 (5元)",
}


def get_prize_level(red_hits: int, blue_hit: int) -> str:
    """根据红球命中数和蓝球命中判定中奖等级。

    Args:
        red_hits: 红球命中数 (0-6)。
        blue_hit: 蓝球命中 (0/1)。

    Returns:
        中奖等级描述字符串。
    """
    return PRIZE_RULES.get((red_hits, blue_hit), f"未中奖 ({red_hits}+{blue_hit})")


def compare_prediction(pred_period: str, config=None) -> dict:
    """对比指定期号的预测结果与实际开奖。

    从 prediction_log 取预测，从 draw_history 取实际开奖，
    计算命中数和等级，写入 result_compare 表。

    Args:
        pred_period: 预测对应的期号（如 "2026082"）。
        config: 模型配置（可选，用于获取 db_path）。

    Returns:
        dict: {
            "period": 期号,
            "status": "ok" / "no_prediction" / "no_draw" / "already_compared",
            "pred_reds": [预测红球],
            "actual_reds": [实际红球],
            "pred_blue": 预测蓝球,
            "actual_blue": 实际蓝球,
            "red_hits": 红球命中数,
            "blue_hit": 蓝球命中 0/1,
            "hit_details": 具体命中的红球号码列表,
            "prize_level": 中奖等级描述,
        }
    """
    # 获取数据库路径
    if config is not None:
        db_path = config.db_path
    else:
        from ssq_sync.config import get_settings
        db_path = get_settings().db_path

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # ── 1. 检查是否已经对比过 ──
        existing = conn.execute(
            "SELECT 1 FROM result_compare WHERE period = ?",
            (pred_period,),
        ).fetchone()
        if existing:
            logger.info(f"期号 {pred_period} 已对比过，跳过")
            # 返回已有结果
            row = conn.execute(
                "SELECT * FROM result_compare WHERE period = ?",
                (pred_period,),
            ).fetchone()
            return {
                "period": pred_period,
                "status": "already_compared",
                "pred_red_hits": row["pred_red_hits"],
                "pred_blue_hit": row["pred_blue_hit"],
                "red_hit_details": row["red_hit_details"],
                "prize_level": row["prize_level"],
            }

        # ── 2. 获取预测 ──
        pred_row = conn.execute(
            """SELECT * FROM prediction_log
               WHERE period = ? ORDER BY id DESC LIMIT 1""",
            (pred_period,),
        ).fetchone()

        if not pred_row:
            logger.warning(f"期号 {pred_period} 无预测记录")
            return {"period": pred_period, "status": "no_prediction"}

        pred_reds = [
            pred_row["red1"], pred_row["red2"], pred_row["red3"],
            pred_row["red4"], pred_row["red5"], pred_row["red6"],
        ]
        pred_blue = pred_row["blue"]

        # ── 3. 获取实际开奖 ──
        draw_row = conn.execute(
            """SELECT * FROM draw_history WHERE period = ?""",
            (pred_period,),
        ).fetchone()

        if not draw_row:
            logger.warning(f"期号 {pred_period} 无开奖记录（可能尚未开奖）")
            return {"period": pred_period, "status": "no_draw"}

        actual_reds = [
            draw_row["red1"], draw_row["red2"], draw_row["red3"],
            draw_row["red4"], draw_row["red5"], draw_row["red6"],
        ]
        actual_blue = draw_row["blue"]

        # ── 4. 计算命中 ──
        pred_set = set(pred_reds)
        actual_set = set(actual_reds)
        hit_reds = sorted(pred_set & actual_set)
        red_hits = len(hit_reds)
        blue_hit = 1 if pred_blue == actual_blue else 0

        # ── 5. 判定等级 ──
        prize_level = get_prize_level(red_hits, blue_hit)

        # ── 6. 写入数据库 ──
        conn.execute(
            """INSERT INTO result_compare
               (period, pred_red_hits, pred_blue_hit, red_hit_details, prize_level)
               VALUES (?, ?, ?, ?, ?)""",
            (
                pred_period, red_hits, blue_hit,
                ",".join(f"{r:02d}" for r in hit_reds) if hit_reds else "",
                prize_level,
            ),
        )
        conn.commit()

        logger.info(
            f"对比完成: 期号={pred_period}, "
            f"红球命中 {red_hits}/6 {hit_reds}, 蓝球{'中' if blue_hit else '不中'}, "
            f"等级={prize_level}"
        )

        return {
            "period": pred_period,
            "status": "ok",
            "pred_reds": pred_reds,
            "actual_reds": actual_reds,
            "pred_blue": pred_blue,
            "actual_blue": actual_blue,
            "red_hits": red_hits,
            "blue_hit": blue_hit,
            "hit_details": [f"{r:02d}" for r in hit_reds],
            "prize_level": prize_level,
            "draw_date": draw_row["draw_date"],
        }

    finally:
        conn.close()


def compare_latest_uncompared(config=None) -> list[dict]:
    """对比所有未对比的预测（有多期待对比时批量处理）。

    Returns:
        对比结果列表。
    """
    if config is not None:
        db_path = config.db_path
    else:
        from ssq_sync.config import get_settings
        db_path = get_settings().db_path

    conn = sqlite3.connect(db_path)

    try:
        # 查找有预测但未对比的期号
        rows = conn.execute(
            """SELECT DISTINCT p.period FROM prediction_log p
               WHERE p.period NOT IN (SELECT period FROM result_compare)
               AND p.period IN (SELECT period FROM draw_history)
               ORDER BY p.period"""
        ).fetchall()

        periods = [r[0] for r in rows]
    finally:
        conn.close()

    if not periods:
        logger.info("没有待对比的预测")
        return []

    logger.info(f"发现 {len(periods)} 期待对比: {periods}")

    results = []
    for period in periods:
        result = compare_prediction(period, config)
        results.append(result)

    return results


# ── CLI 入口 ──
def main():
    """手动触发结果对比（供 pipeline 调用）。"""
    import sys
    results = compare_latest_uncompared()

    if not results:
        print("没有待对比的预测")
        return 0

    for r in results:
        if r["status"] == "ok":
            print(
                f"期号 {r['period']}: "
                f"红球 {r['red_hits']}/6 {r.get('hit_details', [])}, "
                f"蓝球{'中' if r['blue_hit'] else '不中'}, "
                f"{r['prize_level']}"
            )
        else:
            print(f"期号 {r['period']}: {r['status']}")

    return 0


if __name__ == "__main__":
    main()
