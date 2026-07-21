#!/usr/bin/env python
"""双色球统计分析主入口。

三个分析维度，替代原 LSTM 模型预测:
  1. 偏差检测 — 统计检验甄别摇奖机是否存在物理偏差
  2. 覆盖策略 — 组合设计轮选号码推荐
  3. 蓝球分析 — 频率+遗漏+趋势，给出下一期蓝球推荐

管线中调用: python -m ssq_analysis.main
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import date
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from ssq_sync.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "ssq_history.db"


# ═══════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════

def _load_data() -> tuple[np.ndarray, np.ndarray]:
    """加载历史数据。返回 (reds: (n,6), blue: (n,))。"""
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT red1,red2,red3,red4,red5,red6,blue FROM draw_history ORDER BY period"
    ).fetchall()
    conn.close()
    reds = np.array([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows])
    blue = np.array([r[6] for r in rows])
    return reds, blue


# ═══════════════════════════════════════════════════════════════
#  模块1: 物理偏差检测
# ═══════════════════════════════════════════════════════════════

def bias_detection(reds: np.ndarray, blue: np.ndarray) -> str:
    """对历史开奖数据进行多项统计检验，甄别摇奖机物理偏差。

    检验项:
      - 红球频率卡方检验: 各号码出现频率是否均匀
      - 蓝球频率卡方检验: 同红球
      - 游程检验: 序列中位数上下的随机性
      - 自相关检验: 各期号码是否存在时序依赖

    Returns:
        格式化分析报告文本。
    """
    n_draws = len(reds)
    lines = []
    lines.append("【分析一】物理偏差检测")
    lines.append(f"数据: {n_draws} 期")
    lines.append("")

    # ── 1a. 红球频率卡方检验 ──
    red_counts = np.bincount(reds.flatten(), minlength=34)[1:]  # 去掉索引0
    # 理论期望: 每期6个红球, 每个号码期望 = n*6/33
    expected_red = np.full(33, n_draws * 6 / 33)
    chi2_red, p_red = sp_stats.chisquare(red_counts, f_exp=expected_red)

    lines.append("📊 红球频率卡方检验")
    lines.append(f"  χ² = {chi2_red:.2f},  p = {p_red:.4f}")
    if p_red < 0.01:
        lines.append("  ⚠️  极显著偏差! 摇奖机可能存在物理缺陷!")
    elif p_red < 0.05:
        lines.append("  ⚠️  显著偏差, 建议持续关注")
    else:
        lines.append("  ✅ 未检测到显著偏差 (p>0.05)")

    # 异常频率号码
    deviations = (red_counts - expected_red) / np.sqrt(expected_red)
    top_anomalies = np.argsort(np.abs(deviations))[-5:][::-1]
    lines.append("  偏差最大的5个号码:")
    for idx in top_anomalies:
        num = idx + 1
        sign = "偏多↑" if deviations[idx] > 0 else "偏少↓"
        lines.append(f"    {num:02d}: 理论{expected_red[idx]:.0f}, 实际{red_counts[idx]}, {sign} (|z|={abs(deviations[idx]):.2f})")

    lines.append("")

    # ── 1b. 蓝球频率卡方检验 ──
    blue_counts = np.bincount(blue, minlength=17)[1:]
    expected_blue = np.full(16, n_draws / 16)
    chi2_blue, p_blue = sp_stats.chisquare(blue_counts, f_exp=expected_blue)

    lines.append("📊 蓝球频率卡方检验")
    lines.append(f"  χ² = {chi2_blue:.2f},  p = {p_blue:.4f}")
    if p_blue < 0.01:
        lines.append("  ⚠️  极显著偏差!")
    elif p_blue < 0.05:
        lines.append("  ⚠️  显著偏差!")
    else:
        lines.append("  ✅ 未检测到显著偏差")

    lines.append("")

    # ── 1c. 游程检验 ──
    lines.append("📊 游程检验 (中位数上下随机性)")
    for i, pos_name in enumerate(["位置1", "位置2", "位置3", "位置4", "位置5", "位置6"]):
        series = reds[:, i]
        median = np.median(series)
        above = series > median
        runs = 1
        for j in range(1, len(above)):
            if above[j] != above[j-1]:
                runs += 1
        n1, n2 = above.sum(), len(above) - above.sum()
        exp_runs = 2*n1*n2/(n1+n2) + 1 if n1+n2 > 0 else 0
        std_runs = np.sqrt((exp_runs-1)*(exp_runs-2)/(n1+n2-1)) if n1+n2 > 1 else 1
        z = (runs - exp_runs)/std_runs if std_runs > 0 else 0
        p = 2*(1-sp_stats.norm.cdf(abs(z)))
        flag = " ⚠️" if p < 0.05 else ""
        lines.append(f"  {pos_name}: runs={runs} exp={exp_runs:.1f} Z={z:+.2f} p={p:.3f}{flag}")

    lines.append("")

    # ── 1d. 自相关检验 ──
    lines.append("📊 自相关检验 (lag=1)")
    any_sig = False
    for i in range(6):
        series = reds[:, i]
        ac = np.corrcoef(series[1:], series[:-1])[0, 1] if len(series) > 1 else 0
        se = 1/np.sqrt(n_draws)  # standard error under null
        z = ac / se if se > 0 else 0
        p = 2*(1-sp_stats.norm.cdf(abs(z)))
        flag = " ⚠️ 显著!" if p < 0.05 else ""
        if p < 0.05:
            any_sig = True
        lines.append(f"  位置{i+1}: AC={ac:+.4f} p={p:.3f}{flag}")

    lines.append("")
    lines.append("💡 解读:")
    if not any_sig:
        lines.append("  未发现任何统计学显著的自相关，")
        lines.append("  双色球开奖符合'纯随机'假设。")
    else:
        lines.append("  发现显著自相关! 建议进一步核查。")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  模块2: 组合覆盖策略 (轮选号码)
# ═══════════════════════════════════════════════════════════════

def coverage_strategy(reds: np.ndarray, n_pick: int = 15) -> str:
    """基于频率统计+遗漏分析，推荐轮选号码组合。

    策略:
      1. 综合评分 = 频率分 + 遗漏分 + 热度分
      2. 取前 n_pick 个号码作为"胆码池"
      3. 从这个池中可以组成多注，扩大覆盖面

    Args:
        reds: (n, 6) 红球历史数组。
        n_pick: 推荐号码数量 (默认15个)。

    Returns:
        格式化推荐文本。
    """
    n_draws = len(reds)
    lines = []
    lines.append("【分析二】组合覆盖策略 — 轮选号码推荐")
    lines.append("")

    # 1. 频率分: 全局出现次数 / 总次数
    freq = np.bincount(reds.flatten(), minlength=34)[1:].astype(float)
    freq_score = freq / freq.sum()

    # 2. 遗漏分: 最近N期内"冷号"加分
    recent_window = min(30, n_draws)
    recent = reds[-recent_window:]
    recent_set = set(recent.flatten())
    omit_score = np.zeros(33)
    for i in range(33):
        num = i + 1
        if num not in recent_set:
            omit_score[i] = 1.0  # 最近遗漏的号码：可能是"该出了"
        # 热度加权：越久没出分数越高
        for j in range(n_draws-1, -1, -1):
            if num in reds[j]:
                gap = n_draws - 1 - j
                omit_score[i] = min(gap / 30.0, 1.0)
                break

    # 3. 综合评分
    composite = freq_score * 0.4 + omit_score * 0.4 + np.random.RandomState(42).rand(33) * 0.2
    ranking = np.argsort(composite)[::-1] + 1  # 1-based

    lines.append(f"🔢 推荐胆码池 ({n_pick}个号码)")
    lines.append(f"  {' '.join(f'{n:02d}' for n in sorted(ranking[:n_pick]))}")
    lines.append("")
    lines.append("📋 评分依据:")
    lines.append("  - 历史频率 (40%): 出现次数多的号码更可能再次出现")
    lines.append("  - 遗漏热度 (40%): 长期未出的号码可能反弹")
    lines.append("  - 随机扰动 (20%): 避免完全确定性的推荐")

    # 给出几组推荐组合 (从胆码池中按规则选6个)
    lines.append("")
    lines.append("🎯 推荐组合 (从胆码池中选6注):")
    pool = sorted(ranking[:n_pick])
    rng = np.random.RandomState(42)
    for combo_idx in range(6):
        # 从池中随机抽取6个不重复, 递增排序
        chosen = sorted(rng.choice(pool, size=6, replace=False).tolist())
        lines.append(f"  第{combo_idx+1}注: {' '.join(f'{n:02d}' for n in chosen)}")

    lines.append("")
    lines.append("💡 说明: 轮选策略不能提高单注中奖概率,")
    lines.append("  但能保证: 只要开奖号码在此胆码池中,")
    lines.append(f"  你就有一定覆盖率 (6/{n_pick}={6/n_pick:.0%})。")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  模块3: 蓝球统计分析
# ═══════════════════════════════════════════════════════════════

def blue_analysis(blue: np.ndarray) -> str:
    """蓝球统计分析: 频率+遗漏+冷热+趋势。

    Args:
        blue: (n,) 蓝球历史数组。

    Returns:
        格式化分析报告。
    """
    n_draws = len(blue)
    lines = []
    lines.append("【分析三】蓝球统计分析")
    lines.append(f"数据: {n_draws} 期")
    lines.append("")

    # 1. 频率排名
    freq = np.bincount(blue, minlength=17)[1:]
    ranking = np.argsort(freq)[::-1] + 1  # 按频率降序

    lines.append("📊 蓝球历史频率排名")
    for rank, num in enumerate(ranking, 1):
        bar = "█" * int(freq[num-1] / freq.max() * 20)
        lines.append(f"  {rank:2d}. {num:02d} 出现{freq[num-1]:4d}次 ({freq[num-1]/n_draws*100:5.1f}%) {bar}")

    # 2. 最近遗漏
    lines.append("")
    lines.append("📋 蓝球遗漏分析")
    last_seen = {}
    for num in range(1, 17):
        occurrences = np.where(blue == num)[0]
        if len(occurrences) > 0:
            gap = n_draws - 1 - occurrences[-1]
            last_seen[num] = gap
        else:
            last_seen[num] = n_draws

    sorted_by_gap = sorted(last_seen.items(), key=lambda x: x[1], reverse=True)
    for num, gap in sorted_by_gap[:8]:
        tag = " 🔥冷号" if gap > 30 else (" ❄️热号" if gap < 5 else "")
        lines.append(f"  {num:02d}: 遗漏{gap:4d}期{tag}")

    # 3. 趋势分析
    lines.append("")
    lines.append("📈 近50期蓝球趋势")
    recent = blue[-50:]
    # 奇偶
    odd_pct = np.mean(recent % 2 == 1) * 100
    # 大小 (1-8小, 9-16大)
    big_pct = np.mean(recent >= 9) * 100
    lines.append(f"  奇偶比: 奇{odd_pct:.0f}% / 偶{100-odd_pct:.0f}%")
    lines.append(f"  大小比: 大{big_pct:.0f}% / 小{100-big_pct:.0f}%")

    # 4. 综合推荐
    lines.append("")
    lines.append("🎯 下期蓝球推荐 (综合评分)")
    scores = np.zeros(16)
    for i in range(16):
        num = i + 1
        # 频率分
        scores[i] += freq[i] / freq.sum() * 0.4
        # 遗漏分 (遗漏越久分越高)
        if last_seen[num] > 10:
            scores[i] += min(last_seen[num] / 50.0, 1.0) * 0.3
        # 近期热度分 (最近出现过加分)
        if num in recent[-10:]:
            scores[i] += 0.3

    top5 = np.argsort(scores)[::-1][:5] + 1
    lines.append("  Top-5: " + " ".join(
        f"{n:02d}({scores[n-1]:.2f})" for n in top5
    ))

    # 4. 下期蓝球预测
    lines.append("")
    lines.append("🔮 下期蓝球预测")
    # 用频率加权随机采样 (不用argmax, 避免永远选同一个号码)
    probs = np.zeros(16)
    for i in range(16):
        probs[i] = scores[i]
    probs = probs / probs.sum()
    rng = np.random.RandomState(int(date.today().strftime("%Y%m%d")))
    predicted = rng.choice(16, size=1, p=probs)[0] + 1
    lines.append(f"  推荐蓝球: {predicted:02d}")
    lines.append(f"  备选: " + " ".join(f"{n:02d}" for n in top5[:3] if n != predicted))

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════

def run_analysis() -> str:
    """运行完整三项分析, 返回格式化的完整报告文本。

    Returns:
        完整的分析报告字符串。
    """
    reds, blue = _load_data()
    n_draws = len(reds)
    latest_period = str(n_draws)

    parts = []
    parts.append(f"双色球统计分析报告")
    parts.append(f"日期: {date.today().isoformat()} | 数据: 共 {n_draws} 期")
    parts.append("")
    parts.append("⚠️ 说明: 本分析基于统计方法, 不代表能预测未来号码。双色球每期独立开奖, 任何分析方法均无法提高单期中奖概率。本报告仅供研究参考。")
    parts.append("")

    # 三项分析
    parts.append(bias_detection(reds, blue))
    parts.append("")
    parts.append(coverage_strategy(reds))
    parts.append("")
    parts.append(blue_analysis(blue))

    return "\n".join(parts)


def main():
    """CLI 入口, 用于管线调用。"""
    report = run_analysis()
    print(report)
    logger.info("统计分析完成")


if __name__ == "__main__":
    main()
