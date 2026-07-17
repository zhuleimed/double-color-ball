"""报告文本生成模块。

生成日报/周报/月报文本，用于 WxPusher 推送和日志记录。

对标 019_etf_daily_sync_and_backtest/simulation/framework/summary.py 的格式风格。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from ssq_report.tracker import SuccessTracker
from ssq_sync.engine import DataEngine
from ssq_sync.logger import get_logger

logger = get_logger(__name__)


def generate_daily_report(
    compare_result: dict | None = None,
    next_prediction: dict | None = None,
    tracker: SuccessTracker | None = None,
) -> str:
    """生成每日预测报告文本。

    Args:
        compare_result: 上期预测vs实际对比结果（来自 compare.py）。
        next_prediction: 下期预测号码（来自 predict.py）。
        tracker: 成功率追踪器。

    Returns:
        格式化的多行文本。
    """
    today = date.today()
    draw_day_names = {1: "周二", 3: "周四", 6: "周日"}
    weekday_name = draw_day_names.get(today.weekday(), ["一","二","三","四","五","六","日"][today.weekday()])

    lines = []
    lines.append(f"🔴 双色球预测日报 | {today.strftime('%Y年%m月%d日')}({weekday_name})")
    lines.append("═" * 38)

    # ── 第一部分：上期回顾 ──
    if compare_result and compare_result.get("status") == "ok":
        lines.append("")
        lines.append(f"📅 回顾: 第{compare_result['period']}期")
        if compare_result.get("draw_date"):
            lines.append(f"   开奖日期: {compare_result['draw_date']}")

        # 预测号码
        pred_reds = compare_result.get("pred_reds", [])
        pred_blue = compare_result.get("pred_blue", 0)
        reds_str = " ".join(f"{r:02d}" for r in pred_reds) if pred_reds else "—"
        lines.append(f"")
        lines.append(f"🎯 预测号码:")
        lines.append(f"   红球: {reds_str}")
        lines.append(f"   蓝球: {pred_blue:02d}" if pred_blue else "   蓝球: —")

        # 实际号码
        actual_reds = compare_result.get("actual_reds", [])
        actual_blue = compare_result.get("actual_blue", 0)
        actual_reds_str = " ".join(f"{r:02d}" for r in actual_reds) if actual_reds else "—"
        lines.append(f"")
        lines.append(f"🏆 实际开奖:")
        lines.append(f"   红球: {actual_reds_str}")
        lines.append(f"   蓝球: {actual_blue:02d}" if actual_blue else "   蓝球: —")

        # 对比结果
        red_hits = compare_result.get("red_hits", 0)
        blue_hit = compare_result.get("blue_hit", 0)
        hit_details = compare_result.get("hit_details", [])
        prize = compare_result.get("prize_level", "—")

        hit_bar = _hit_bar(red_hits)
        lines.append(f"")
        lines.append(f"📊 对比结果:")
        lines.append(f"   红球命中: {red_hits}/6 {hit_bar}")
        if hit_details:
            lines.append(f"   命中号码: {' '.join(hit_details)}")
        lines.append(f"   蓝球命中: {'✅ 中' if blue_hit else '❌ 不中'}")
        if not blue_hit and actual_blue and pred_blue:
            lines.append(f"   (预测{pred_blue:02d} 实际{actual_blue:02d})")
        lines.append(f"   中奖等级: {prize}")

    elif compare_result and compare_result.get("status") == "no_draw":
        lines.append("")
        lines.append(f"📅 第{compare_result['period']}期: 尚未开奖，等待结果...")
    else:
        lines.append("")
        lines.append("📅 无上期对比数据")

    # ── 第二部分：累积统计 ──
    if tracker:
        stats = tracker.get_stats()
        total = stats.get("total_predictions", 0)
        if total > 0:
            lines.append("")
            lines.append("─" * 38)
            lines.append(f"📈 累积统计 (共{total}期预测):")
            lines.append("")

            rates = stats.get("red_hit_rates", {})
            # 红球命中分布
            for r in range(7):
                pct = rates.get(f"red_{r}", 0)
                bar = _pct_bar(pct, 10)
                label = f"红球{r}/6" if r > 0 else "红球0/6"
                lines.append(f"   {label:<8} {bar} {pct:.1%}")

            # 汇总指标
            lines.append("")
            lines.append(f"   红球≥3: {stats.get('red_3plus_rate', 0):.1%}")
            lines.append(f"   蓝球中: {stats.get('blue_hit_rate', 0):.1%}")
            lines.append(f"   中奖率: {stats.get('any_prize_rate', 0):.1%}")

            # 最佳记录
            best = stats.get("best_record")
            if best:
                lines.append(f"")
                lines.append(f"   🏅 最佳: 第{best['period']}期 {best['prize']}")

            # 滚动统计
            rolling = stats.get("rolling", {})
            rolling_20 = rolling.get("last_20", {})
            if rolling_20.get("n", 0) > 0:
                lines.append(f"")
                lines.append(f"   近{rolling_20['n']}期: 红≥3={rolling_20['red_3plus_pct']:.1%} "
                             f"蓝中={rolling_20['blue_hit_pct']:.1%}")

    # ── 第三部分：下期预测 ──
    if next_prediction:
        lines.append("")
        lines.append("─" * 38)
        lines.append(f"🔮 下期预测 (第{next_prediction.get('period', '?')}期):")
        lines.append("")
        reds_str = " ".join(f"{r:02d}" for r in next_prediction.get("reds", []))
        lines.append(f"   红球: {reds_str}")
        lines.append(f"   蓝球: {next_prediction.get('blue', 0):02d}")

        if next_prediction.get("blue_top3"):
            top3 = " ".join(f"{b:02d}" for b in next_prediction["blue_top3"])
            lines.append(f"   蓝球备选: {top3}")

    # ── 底部 ──
    lines.append("")
    lines.append("═" * 38)
    lines.append(f"🕐 {datetime.now().strftime('%H:%M')}")

    return "\n".join(lines)


def _hit_bar(hits: int, width: int = 6) -> str:
    """生成红球命中可视条。"""
    filled = "🟢" * hits
    empty = "⚪" * (width - hits)
    return filled + empty


def _pct_bar(pct: float, width: int = 10) -> str:
    """生成百分比条形图。"""
    filled = int(round(pct * width))
    return "█" * filled + "░" * (width - filled)


def generate_summary_text(
    compare_result: dict | None = None,
    next_prediction: dict | None = None,
    tracker_path: str | None = None,
) -> str:
    """便捷函数：一键生成日报文本。

    Args:
        compare_result: 对比结果。
        next_prediction: 下期预测。
        tracker_path: tracker JSON 路径。

    Returns:
        格式化的日报文本。
    """
    tracker = SuccessTracker(tracker_path) if tracker_path else SuccessTracker()
    return generate_daily_report(compare_result, next_prediction, tracker)
