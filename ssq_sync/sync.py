"""数据同步模块：双色球开奖数据的全量/增量同步。

数据流向：
  - cwl.gov.cn API（主）→ akshare（备）→ zhcw.com（兜底）→ draw_history 表
  - 同步日志写入 sync_log 表

管线流程：
  1. 检查数据源可用性 → 三源降级获取数据
  2. 全量回填模式：从 start_date 起全量拉取
  3. 增量模式：检查最新期号 → 仅获取新数据
  4. 写入数据库 → 记录 sync_log

对标 019_etf_daily_sync_and_backtest/etf_sync/sync.py 的 ETFSync 模式。
"""

from __future__ import annotations

import time as time_module
from datetime import date, datetime
from typing import Optional

from ssq_sync.config import Settings, get_settings
from ssq_sync.data_source import fetch_all_draws
from ssq_sync.engine import DataEngine
from ssq_sync.logger import get_logger

logger = get_logger(__name__)


class SSQSync:
    """双色球数据同步管理器。

    管理数据源生命周期、全量/增量同步策略、同步日志记录。

    用法:
        sync_mgr = SSQSync(settings)
        result = sync_mgr.run_full()       # 标准管线
        result = sync_mgr.backfill()       # 全量回填
        result = sync_mgr.sync_latest()    # 仅增量同步最新
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """初始化 SSQSync。

        Args:
            settings: 系统配置（db_path, start_date 等）。
        """
        self.settings = settings or get_settings()
        self.engine = DataEngine(self.settings)
        # 确保数据库表存在
        self.engine.initialize()

    # ════════════════════════════════════════════════════════
    #  开奖日判断
    # ════════════════════════════════════════════════════════

    @staticmethod
    def is_draw_day(check_date: date | None = None) -> bool:
        """判断指定日期是否为双色球开奖日（每周二/四/日）。

        Args:
            check_date: 待检查日期，默认今天。

        Returns:
            True 表示开奖日。
        """
        if check_date is None:
            check_date = date.today()

        # weekday(): 周一=0, 周二=1, 周三=2, 周四=3, 周五=4, 周六=5, 周日=6
        is_draw = check_date.weekday() in (1, 3, 6)
        logger.info(
            f"is_draw_day: {check_date.isoformat()} "
            f"({'开奖日' if is_draw else '非开奖日'})"
        )
        return is_draw

    # ════════════════════════════════════════════════════════
    #  核心同步逻辑
    # ════════════════════════════════════════════════════════

    def backfill(self, start_date: str | None = None,
                 end_date: str | None = None) -> dict:
        """全量回填历史数据。

        从 start_date（或配置中的默认值）起，拉取全部历史开奖数据，
        写入数据库。使用 INSERT OR IGNORE 防重复。

        Args:
            start_date: 起始日期 "YYYY-MM-DD"，None 用配置默认值。
            end_date: 结束日期 "YYYY-MM-DD"，None 表示到今天。

        Returns:
            dict: {"status": "ok"/"error", "inserted": N, "total_fetched": N,
                   "source": "cwl"/"akshare"/"zhcw", "duration_seconds": N}
        """
        if start_date is None:
            start_date = self.settings.start_date
        if end_date is None:
            end_date = date.today().isoformat()

        t0 = time_module.time()
        logger.info(f"=== 全量回填: {start_date} → {end_date} ===")

        # 1. 获取数据
        draws, source = fetch_all_draws(
            start_date=start_date, end_date=end_date, include_history=True
        )

        if not draws:
            elapsed = time_module.time() - t0
            self._log_sync("failed", 0, "none", elapsed, "所有数据源均失败")
            return {
                "status": "error",
                "inserted": 0,
                "total_fetched": 0,
                "source": "none",
                "duration_seconds": elapsed,
                "error": "所有数据源均无法获取数据",
            }

        # 2. 按期号排序（确保数据库中顺序正确）
        draws.sort(key=lambda d: d["period"])

        # 3. 写入数据库
        inserted = self.engine.insert_draws(draws)
        elapsed = time_module.time() - t0

        total_count = self.engine.get_total_count()

        # 4. 记录同步日志
        self._log_sync(
            status="success" if inserted >= 0 else "partial",
            new_draws=inserted,
            source=source,
            duration=elapsed,
            total_count=total_count,
        )

        logger.info(
            f"回填完成: 获取 {len(draws)} 条, 新增 {inserted} 条, "
            f"数据库共 {total_count} 条, 耗时 {elapsed:.0f}s"
        )

        return {
            "status": "ok",
            "inserted": inserted,
            "total_fetched": len(draws),
            "total_in_db": total_count,
            "source": source,
            "duration_seconds": elapsed,
        }

    def sync_latest(self, force: bool = False) -> dict:
        """增量同步：仅获取数据库中缺失的最新开奖数据。

        检查数据库中最新的期号，从官方 API 获取最新数据，
        如果有新期号则插入。

        Args:
            force: 强制模式，跳过数据已是最新检查。

        Returns:
            dict: {"status": "ok"/"skipped"/"error", "new_draws": N, ...}
        """
        t0 = time_module.time()
        logger.info("=== 增量同步 ===")

        if not force:
            latest = self.engine.get_latest_period()
            if latest:
                logger.info(f"当前数据库最新期号: {latest}")

        # 增量模式：仅从 cwl.gov.cn 获取最近 2 页（200 条），高效检查新数据
        from ssq_sync.data_source import fetch_cwl

        draws = fetch_cwl(max_pages=2)  # 只取最新200条
        source = "cwl"

        if not draws:
            # 降级到 zhcw.com
            logger.info("cwl.gov.cn 失败，降级到 zhcw.com...")
            from ssq_sync.data_source import fetch_zhcw
            draws = fetch_zhcw()
            source = "zhcw" if draws else "none"

        if not draws:
            elapsed = time_module.time() - t0
            self._log_sync("failed", 0, "none", elapsed, "无法获取数据")
            return {
                "status": "error",
                "new_draws": 0,
                "source": "none",
                "duration_seconds": elapsed,
                "error": "所有数据源均无法获取数据",
            }

        # 按期号排序，只保留比数据库中最新的还新的
        draws.sort(key=lambda d: d["period"])
        latest_db = self.engine.get_latest_period()

        if latest_db:
            new_draws_list = [d for d in draws if d["period"] > latest_db]
        else:
            new_draws_list = draws  # 数据库为空，全部插入

        if not new_draws_list:
            elapsed = time_module.time() - t0
            logger.info("数据已是最新，无需同步")
            self._log_sync("skipped", 0, source, elapsed)
            return {
                "status": "skipped",
                "new_draws": 0,
                "source": source,
                "duration_seconds": elapsed,
            }

        # 写入
        inserted = self.engine.insert_draws(new_draws_list)
        elapsed = time_module.time() - t0
        total_count = self.engine.get_total_count()

        self._log_sync(
            status="success",
            new_draws=inserted,
            source=source,
            duration=elapsed,
            total_count=total_count,
        )

        latest_new = new_draws_list[-1]["period"]
        logger.info(
            f"增量同步完成: 新增 {inserted} 期, "
            f"最新期号 {latest_new}, 耗时 {elapsed:.0f}s"
        )

        return {
            "status": "ok",
            "new_draws": inserted,
            "latest_period": latest_new,
            "total_in_db": total_count,
            "source": source,
            "duration_seconds": elapsed,
        }

    def run_full(self) -> dict:
        """标准全管线同步（首次使用或需要完整同步时）。

        步骤: 检查数据库状态 → 增量同步 → 如果差距大则触发回填。

        Returns:
            dict: 包含各阶段状态和结果的字典。
        """
        t0 = time_module.time()

        total_count = self.engine.get_total_count()
        latest = self.engine.get_latest_period()

        logger.info(f"数据库状态: {total_count} 期, 最新: {latest or '无'}")

        # 如果数据库为空或数据很少，先做回填
        if total_count < 100:
            logger.info("数据库数据不足，执行全量回填...")
            result = self.backfill()
        else:
            # 增量同步
            result = self.sync_latest()

        elapsed = time_module.time() - t0
        result["duration_seconds"] = elapsed
        return result

    # ── 内部方法 ──

    def _log_sync(self, status: str, new_draws: int, source: str,
                  duration: float, total_count: int | None = None,
                  error: str | None = None) -> None:
        """写入 sync_log 记录。"""
        if total_count is None:
            total_count = self.engine.get_total_count()
        self.engine.insert_sync_log({
            "date": date.today().isoformat(),
            "status": status,
            "new_draws": new_draws,
            "total_draws": total_count,
            "source": source,
            "duration_seconds": round(duration, 1),
            "error_msg": error,
        })
