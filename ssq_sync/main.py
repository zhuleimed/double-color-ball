"""双色球数据同步 — 主程序入口。

对标 019_etf_daily_sync_and_backtest/main.py 的 CLI 模式。

运行模式：
  python -m ssq_sync.main                    # 标准模式：增量同步
  python -m ssq_sync.main --sync-only        # 仅增量同步最新开奖
  python -m ssq_sync.main --force            # 强制模式（跳过最新检查）
  python -m ssq_sync.main --backfill         # 全量回填：从 start_date 起拉取全部历史
  python -m ssq_sync.main --backfill --start 2020-01-01  # 指定回填起始日期

首次使用：python -m ssq_sync.main --backfill
日常使用：python -m ssq_sync.main --sync-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── 确保项目根目录在 path 中 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from ssq_sync.config import get_settings
from ssq_sync.logger import get_logger
from ssq_sync.notify import (
    push_backfill_summary,
    push_error_alert,
    push_sync_summary,
)
from ssq_sync.sync import SSQSync


def main() -> None:
    parser = argparse.ArgumentParser(
        description="🔴 双色球开奖数据同步",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m ssq_sync.main --backfill           # 全量回填所有历史数据
  python -m ssq_sync.main --sync-only          # 增量同步最新开奖
  python -m ssq_sync.main --backfill --start 2020-01-01  # 从指定日期回填
  python -m ssq_sync.main --force              # 强制同步(跳过检查)
        """,
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="仅执行增量数据同步",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制模式：跳过数据已是最新检查",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="回填模式：全量拉取历史数据（从 start_date 起）",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="回填起始日期 YYYY-MM-DD（默认使用 .env 中的 START_DATE）",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="回填结束日期 YYYY-MM-DD（默认今天）",
    )
    args = parser.parse_args()

    try:
        settings = get_settings()
        logger = get_logger(__name__)
        sync_mgr = SSQSync(settings)
        t0 = time.time()

        if args.backfill:
            # ═══════════════════════════════════════
            #  全量回填模式
            # ═══════════════════════════════════════
            start = args.start or settings.start_date
            logger.info(f"=== 全量回填模式: {start} → {args.end or '今天'} ===")
            result = sync_mgr.backfill(start_date=start, end_date=args.end)
            elapsed = time.time() - t0
            logger.info(
                f"回填完成: status={result['status']}, "
                f"新增 {result.get('inserted', 0)} 期, "
                f"数据源 {result.get('source', 'unknown')}, "
                f"耗时 {elapsed:.0f}s"
            )
            # 推送回填完成通知
            push_backfill_summary(settings, result)
            if result["status"] == "error":
                sys.exit(1)
            return

        if args.sync_only:
            # ═══════════════════════════════════════
            #  增量同步模式
            # ═══════════════════════════════════════
            logger.info("=== 增量同步模式 ===")
            result = sync_mgr.sync_latest(force=args.force)
            elapsed = time.time() - t0
            logger.info(
                f"同步完成: status={result['status']}, "
                f"新增 {result.get('new_draws', 0)} 期, "
                f"耗时 {elapsed:.0f}s"
            )
            # 推送同步摘要
            push_sync_summary(settings, result)
            if result["status"] == "error":
                sys.exit(1)
            return

        # ═══════════════════════════════════════════
        #  标准模式（run_full 管线）
        # ═══════════════════════════════════════════
        logger.info("=== 标准模式 ===")
        result = sync_mgr.run_full()
        elapsed = time.time() - t0
        logger.info(
            f"管线完成: status={result.get('status', 'unknown')}, "
            f"耗时 {elapsed:.0f}s"
        )

        # 根据不同的子结果推送
        if result.get("status") == "ok":
            if result.get("total_fetched", 0) > 10:
                push_backfill_summary(settings, result)
            else:
                push_sync_summary(settings, result)

    except Exception:
        try:
            _logger = get_logger(__name__)
            _logger.exception("主流程未捕获异常，程序终止")
        except Exception:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
