"""日志模块：基于 rich 库提供带颜色的结构化终端日志输出。

对标 019_etf_daily_sync_and_backtest/etf_sync/logger.py，完全一致的模式。
"""

import logging
from rich.logging import RichHandler

_FORMAT = "%(name)s - %(message)s"


def get_logger(name: str) -> logging.Logger:
    """工厂函数，返回配置了 RichHandler 的 Logger 实例。

    Args:
        name: logger 名称，通常传入 __name__。

    Returns:
        logging.Logger: 配置好的 Logger 实例。
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        log_time_format="[%Y-%m-%d %H:%M:%S]",
    )
    handler.setFormatter(logging.Formatter(_FORMAT))

    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    return logger
