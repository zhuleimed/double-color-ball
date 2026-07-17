"""SQLite 数据库引擎：建表、查询、插入双色球开奖数据。

对标 019_etf_daily_sync_and_backtest/etf_sync/engine.py 的 DataEngine 模式。
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from ssq_sync.config import Settings, get_settings
from ssq_sync.logger import get_logger

logger = get_logger(__name__)

# ── 建表 SQL ──
CREATE_DRAW_HISTORY = """
CREATE TABLE IF NOT EXISTS draw_history (
    period TEXT PRIMARY KEY,          -- 期号，如 "2025083"
    draw_date TEXT NOT NULL,          -- 开奖日期 YYYY-MM-DD
    red1 INTEGER NOT NULL,
    red2 INTEGER NOT NULL,
    red3 INTEGER NOT NULL,
    red4 INTEGER NOT NULL,
    red5 INTEGER NOT NULL,
    red6 INTEGER NOT NULL,
    blue INTEGER NOT NULL,
    sales_amount REAL,                -- 销售额（元）
    pool_amount REAL,                 -- 奖池金额（元）
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_SYNC_LOG = """
CREATE TABLE IF NOT EXISTS sync_log (
    date TEXT PRIMARY KEY,            -- 同步日期 YYYY-MM-DD
    status TEXT NOT NULL,             -- success / failed / skipped
    new_draws INTEGER DEFAULT 0,      -- 新增开奖期数
    total_draws INTEGER DEFAULT 0,    -- 数据库总期数
    source TEXT,                      -- 数据来源: cwl / akshare / zhcw
    duration_seconds REAL,            -- 耗时(秒)
    error_msg TEXT                    -- 错误信息
)
"""

# ── 预测记录表（供后续 P4 使用，这里先建好表结构） ──
CREATE_PREDICTION_LOG = """
CREATE TABLE IF NOT EXISTS prediction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    period TEXT NOT NULL,             -- 预测对应的期号
    pred_date TEXT NOT NULL,          -- 预测生成日期
    red1 INTEGER, red2 INTEGER, red3 INTEGER,
    red4 INTEGER, red5 INTEGER, red6 INTEGER,
    blue INTEGER,
    model_version TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

# ── 结果对比表（供 P5 使用，这里先建好） ──
CREATE_RESULT_COMPARE = """
CREATE TABLE IF NOT EXISTS result_compare (
    period TEXT PRIMARY KEY,          -- 期号
    pred_red_hits INTEGER,            -- 红球命中数 0-6
    pred_blue_hit INTEGER,            -- 蓝球命中 0/1
    red_hit_details TEXT,             -- 命中的具体红球号码，逗号分隔
    prize_level TEXT,                 -- 中奖等级描述
    compare_time TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_draw_date ON draw_history(draw_date)",
    "CREATE INDEX IF NOT EXISTS idx_pred_period ON prediction_log(period)",
]


class DataEngine:
    """双色球数据库引擎，封装 SQLite 操作。

    用法:
        engine = DataEngine(settings)
        engine.initialize()  # 首次使用建表
        count = engine.insert_draws(df)
        df = engine.get_draws_df()
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.db_path = Path(self.settings.db_path)
        # 确保父目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 连接管理 ──

    def _connect(self) -> sqlite3.Connection:
        """获取数据库连接。"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── 初始化 ──

    def initialize(self) -> None:
        """建表 + 创建索引（幂等操作，首次使用调用）。"""
        conn = self._connect()
        try:
            conn.execute(CREATE_DRAW_HISTORY)
            conn.execute(CREATE_SYNC_LOG)
            conn.execute(CREATE_PREDICTION_LOG)
            conn.execute(CREATE_RESULT_COMPARE)
            for idx_sql in CREATE_INDEXES:
                conn.execute(idx_sql)
            conn.commit()
            logger.info(f"数据库初始化完成: {self.db_path}")
        finally:
            conn.close()

    # ── 写入 ──

    def insert_draws(self, draws: list[dict]) -> int:
        """批量插入开奖记录，使用 INSERT OR IGNORE 防重复。

        Args:
            draws: 开奖记录列表，每条 dict 需含 period, draw_date, red1-6, blue，
                   可选 sales_amount, pool_amount。

        Returns:
            实际新增的行数。
        """
        if not draws:
            return 0

        conn = self._connect()
        inserted = 0
        try:
            cur = conn.cursor()
            for d in draws:
                try:
                    cur.execute(
                        """INSERT OR IGNORE INTO draw_history
                           (period, draw_date, red1, red2, red3, red4, red5, red6,
                            blue, sales_amount, pool_amount)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            str(d["period"]),
                            str(d["draw_date"]),
                            int(d["red1"]), int(d["red2"]), int(d["red3"]),
                            int(d["red4"]), int(d["red5"]), int(d["red6"]),
                            int(d["blue"]),
                            d.get("sales_amount"),
                            d.get("pool_amount"),
                        ),
                    )
                    if cur.rowcount > 0:
                        inserted += 1
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"跳过无效记录 {d.get('period', '?')}: {e}")
            conn.commit()
        finally:
            conn.close()

        return inserted

    def insert_sync_log(self, entry: dict) -> None:
        """写入同步日志（INSERT OR REPLACE）。"""
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO sync_log
                   (date, status, new_draws, total_draws, source, duration_seconds, error_msg)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.get("date", date.today().isoformat()),
                    entry.get("status", "unknown"),
                    entry.get("new_draws", 0),
                    entry.get("total_draws", 0),
                    entry.get("source", ""),
                    entry.get("duration_seconds", 0),
                    entry.get("error_msg", None),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ── 查询 ──

    def get_latest_period(self) -> Optional[str]:
        """获取数据库中最新一期期号。"""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT period FROM draw_history ORDER BY period DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def get_latest_date(self) -> Optional[str]:
        """获取数据库中最新开奖日期。"""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT draw_date FROM draw_history ORDER BY draw_date DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def get_total_count(self) -> int:
        """获取开奖记录总数。"""
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) FROM draw_history").fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def get_draws_df(self, start_date: str | None = None,
                     end_date: str | None = None) -> pd.DataFrame:
        """查询开奖数据，返回 DataFrame。

        Args:
            start_date: 起始日期 YYYY-MM-DD（含），None 表示不限制。
            end_date: 结束日期 YYYY-MM-DD（含），None 表示不限制。

        Returns:
            DataFrame，列: period, draw_date, red1-6, blue。
        """
        conn = self._connect()
        try:
            sql = """SELECT period, draw_date, red1, red2, red3, red4, red5, red6, blue
                     FROM draw_history WHERE 1=1"""
            params: list = []
            if start_date:
                sql += " AND draw_date >= ?"
                params.append(start_date)
            if end_date:
                sql += " AND draw_date <= ?"
                params.append(end_date)
            sql += " ORDER BY period ASC"
            return pd.read_sql_query(sql, conn, params=params)
        finally:
            conn.close()

    def get_all_draws_df(self) -> pd.DataFrame:
        """获取全部开奖数据，按期号升序排列。"""
        return self.get_draws_df()

    def period_exists(self, period: str) -> bool:
        """检查指定期号是否已存在。"""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM draw_history WHERE period = ?", (str(period),)
            ).fetchone()
            return row is not None
        finally:
            conn.close()
