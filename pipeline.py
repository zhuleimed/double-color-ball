#!/usr/bin/env python
"""双色球预测全自动管线编排器。

由 cron 在开奖日 23:00 和次日 10:00 启动，依次执行：
  夜间 (--night-run): 数据同步 → 结果对比 → 模型预测
  上午 (--push-report): 生成日报 → WxPusher 推送

使用方法：
    python pipeline.py --night-run       # 夜间数据+对比+预测
    python pipeline.py --push-report     # 推送日报
    python pipeline.py --backfill        # 首次使用：全量回填

对标 019_etf_daily_sync_and_backtest/pipeline.py 的完整模式。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

# 强制 stdout/stderr 无缓冲
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── 让 Python 能找到项目包 ──
PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv
load_dotenv()

from pipeline_status import PipelineStatus, push_pipeline_summary
from ssq_report.notify import send_message, push_simple_notice
from ssq_report.reporter import generate_daily_report, generate_summary_text
from ssq_report.tracker import SuccessTracker
from ssq_report.compare import compare_latest_uncompared
from ssq_sync.sync import SSQSync

# ════════════════════════════════════════════════════════════
#  常量
# ════════════════════════════════════════════════════════════

PYTHON = sys.executable

# ════════════════════════════════════════════════════════════
#  开奖日判断
# ════════════════════════════════════════════════════════════

def is_draw_day(check_date: date | None = None) -> bool:
    """判断是否为双色球开奖日（每周二/四/日）。"""
    if check_date is None:
        check_date = date.today()
    return check_date.weekday() in (1, 3, 6)


# ════════════════════════════════════════════════════════════
#  Pipeline 步骤
# ════════════════════════════════════════════════════════════

def run_step(step_id: str, name: str, cmd: list, cwd: str,
             required: bool = True, timeout: int = 3600,
             ps: PipelineStatus | None = None) -> bool:
    """执行一个 pipeline 步骤（子进程）。

    Args:
        step_id: 步骤 ID。
        name: 步骤名称。
        cmd: 命令列表（不含 Python 解释器）。
        cwd: 工作目录。
        required: 是否必需（失败时是否终止管线）。
        timeout: 超时秒数。
        ps: PipelineStatus 实例。

    Returns:
        True 表示成功。
    """
    if ps:
        ps.start_step(step_id)

    print(f"\n  ▶ {name}...")
    t0 = time.time()

    try:
        proc = subprocess.Popen(
            [PYTHON] + cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout_lines, stderr_lines = [], []

        if proc.stdout:
            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip("\n")
                stdout_lines.append(line)
                print(f"    {line}")
        if proc.stderr:
            for line in iter(proc.stderr.readline, ""):
                line = line.rstrip("\n")
                stderr_lines.append(line)
                if "WARNING" in line or "ERROR" in line:
                    print(f"    {line}", file=sys.stderr)

        proc.wait(timeout=timeout if timeout > 0 else None)
        elapsed = time.time() - t0

        success = proc.returncode == 0
        detail = {
            "returncode": proc.returncode,
            "stdout_last": "\n".join(stdout_lines[-10:])[-500:],
            "stderr_last": "\n".join(stderr_lines[-5:])[-300:],
        }

        if ps:
            ps.complete_step(
                step_id, success=success, detail=detail,
                error=None if success else (stderr_lines[-1] if stderr_lines else "未知错误"),
            )

        if success:
            print(f"  ✅ {name} 完成（{elapsed:.0f}s）")
        else:
            print(f"  ❌ {name} 失败（{elapsed:.0f}s）")

        return success

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        print(f"  ⏰ {name} 超时（{timeout}s）")
        if ps:
            ps.complete_step(step_id, success=False, error=f"超时 {timeout}s")
        return False

    except Exception as e:
        print(f"  💥 {name} 异常: {e}")
        if ps:
            ps.complete_step(step_id, success=False, error=str(e))
        return False


# ════════════════════════════════════════════════════════════
#  完整管线（合并夜间+日报 — 每日 10:00 执行）
# ════════════════════════════════════════════════════════════

def run_pipeline():
    """完整管线: 数据同步 → 结果对比 → 模型预测 → 日报推送。

    每日 10:00 触发（开奖日次日更佳）。
    数据在开奖日当晚已可用，10:00 绝对可靠。
    """
    today = date.today()
    today_str = today.isoformat()

    print(f"\n{'=' * 55}")
    print(f"  🔴 双色球预测管线 | {today_str} 星期{today.weekday()+1}")
    print(f"  {'=' * 55}")

    # ── 初始化状态 ──
    ps = PipelineStatus()
    ps.reset()
    pipeline_ok = True

    # ── Step 1: 数据同步（必需） ──
    steps_processing = [
        {"id": "sync", "name": "数据同步", "cmd": ["-m", "ssq_sync.main", "--sync-only"],
         "required": True, "timeout": 300},
        {"id": "compare", "name": "结果对比", "cmd": ["-m", "ssq_report.compare"],
         "required": False, "timeout": 120},
        {"id": "analyze", "name": "统计分析(偏差+覆盖+蓝球)", "cmd": ["-m", "ssq_analysis.main"],
         "required": True, "timeout": 300},
    ]

    for s in steps_processing:
        ps.add_step(s["id"], s["name"])

    for step in steps_processing:
        success = run_step(
            step["id"], step["name"], step["cmd"],
            cwd=str(PROJECT_DIR),
            required=step["required"],
            timeout=step["timeout"],
            ps=ps,
        )
        if not success and step["required"]:
            pipeline_ok = False
            break

    # ── Step 4: 推送统计分析报告 ──
    ps.add_step("report", "报告推送")
    ps.start_step("report")

    try:
        from ssq_analysis.main import run_analysis
        from ssq_report.notify import send_message

        # 生成统计分析报告
        analysis_report = run_analysis()
        print(analysis_report)

        # 推送到微信
        send_message("双色球统计分析报告", analysis_report)
        print("  ✅ 分析报告已推送到微信")

        ps.complete_step("report", success=True)
    except Exception as e:
        print(f"  ❌ 报告推送失败: {e}")
        ps.complete_step("report", success=False, error=str(e))

    # ── 结束 ──
    status = "completed" if pipeline_ok else "failed"
    ps.finish(status)

    try:
        push_pipeline_summary(ps.to_dict(), send_message)
    except Exception as e:
        print(f"  ⚠ 推送管线汇总异常: {e}")

    print(f"\n  {'=' * 55}")
    print(f"  管线状态: {status}")
    print(f"  {'=' * 55}\n")

    sys.exit(0 if pipeline_ok else 1)


# ════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="🔴 双色球预测管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python pipeline.py                   # 完整管线（数据同步→对比→预测→推送）
  python pipeline.py --backfill        # 首次使用：全量回填历史数据
  python pipeline.py --predict-only    # 仅预测（训练后手动触发）
        """,
    )
    parser.add_argument("--backfill", action="store_true",
                        help="全量回填历史数据（首次使用）")
    parser.add_argument("--predict-only", action="store_true",
                        help="仅执行模型预测+推送（不对比）")

    args = parser.parse_args()

    if args.backfill:
        run_step("backfill", "全量回填",
                 ["-m", "ssq_sync.main", "--backfill"],
                 str(PROJECT_DIR), required=True, timeout=600)
        return

    if args.predict_only:
        # 仅预测+推送（用于训练后手动触发）
        run_step("predict", "模型预测",
                 ["-m", "ssq_model.predict", "--beam", "5"],
                 str(PROJECT_DIR), required=True, timeout=300)
        # 推送简单预测
        import sqlite3
        conn = sqlite3.connect(str(PROJECT_DIR / "data" / "ssq_history.db"))
        conn.row_factory = sqlite3.Row
        pred_row = conn.execute("SELECT * FROM prediction_log ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        if pred_row:
            next_prediction = {
                "period": pred_row["period"],
                "reds": [pred_row[f"red{i}"] for i in range(1, 7)],
                "blue": pred_row["blue"],
            }
            tracker = SuccessTracker()
            report = generate_daily_report(None, next_prediction, tracker)
            print(report)
            from ssq_report.notify import push_daily_report
            push_daily_report(report)
        return

    # 默认：完整管线
    run_pipeline()


if __name__ == "__main__":
    main()
