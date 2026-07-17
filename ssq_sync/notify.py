"""消息推送模块：通过 WxPusher 发送微信通知。

对标 019_etf_daily_sync_and_backtest/etf_sync/notify.py 的推送模式。
"""

from __future__ import annotations

from datetime import date, datetime

from ssq_sync.config import Settings
from ssq_sync.logger import get_logger

logger = get_logger(__name__)


def _do_push(settings: Settings, title: str, content: str) -> bool:
    """执行 WxPusher 推送。

    Args:
        settings: 系统配置。
        title: 消息标题（用于日志）。
        content: 消息正文。

    Returns:
        True 表示推送成功。
    """
    if not settings.wxpusher_token:
        logger.info("未配置 WxPusher Token，跳过推送")
        return False

    try:
        from wxpusher import WxPusher

        r = WxPusher.send_message(
            content=content,
            token=settings.wxpusher_token,
            topic_ids=settings.wxpusher_topic_ids,
            content_type=1,  # 纯文本
        )
        if r.get("code") == 1000:
            logger.info(f"推送成功: {title}")
            return True
        else:
            logger.warning(f"推送失败: {r}")
            return False
    except Exception as e:
        logger.warning(f"推送异常: {e}")
        return False


def push_backfill_summary(
    settings: Settings, result: dict
) -> bool:
    """推送全量回填完成通知。

    Args:
        settings: 系统配置。
        result: backfill() 返回的结果字典。
    """
    if not settings.wxpusher_token:
        return False

    today_str = date.today().strftime("%m-%d")
    now_str = datetime.now().strftime("%H:%M")
    status_icon = "✅" if result.get("status") == "ok" else "❌"

    message = (
        f"🔴 双色球数据回填完成 | {today_str}\n\n"
        f"状态: {status_icon} {result.get('status', 'unknown')}\n"
        f"执行: {now_str}\n\n"
        f"📥 获取: {result.get('total_fetched', 0)} 条记录\n"
        f"💾 新增: {result.get('inserted', 0)} 条\n"
        f"📊 数据库总计: {result.get('total_in_db', 0)} 期\n"
        f"📡 数据源: {result.get('source', 'unknown')}\n"
        f"⏱ 耗时: {result.get('duration_seconds', 0):.0f} 秒"
    )

    if result.get("error"):
        message += f"\n\n⚠️ 异常: {result['error']}"

    return _do_push(settings, "双色球数据回填", message)


def push_sync_summary(
    settings: Settings, result: dict
) -> bool:
    """推送增量同步完成通知。

    Args:
        settings: 系统配置。
        result: sync_latest() 返回的结果字典。
    """
    if not settings.wxpusher_token:
        return False

    today_str = date.today().strftime("%m-%d")
    now_str = datetime.now().strftime("%H:%M")
    status = result.get("status", "unknown")

    if status == "skipped":
        message = (
            f"🔴 双色球数据同步 | {today_str}\n\n"
            f"⏭️ 数据已是最新，跳过同步\n"
            f"📊 数据库: {result.get('total_in_db', 0)} 期\n"
            f"{now_str}"
        )
    elif status == "ok":
        message = (
            f"🔴 双色球数据同步完成 | {today_str}\n\n"
            f"✅ 增量同步成功\n"
            f"执行: {now_str}\n\n"
            f"🆕 新增: {result.get('new_draws', 0)} 期\n"
            f"📊 最新期号: {result.get('latest_period', '?')}\n"
            f"📡 数据源: {result.get('source', 'unknown')}\n"
            f"⏱ 耗时: {result.get('duration_seconds', 0):.0f} 秒"
        )
    else:
        message = (
            f"🔴 双色球数据同步失败 | {today_str}\n\n"
            f"❌ 同步异常\n"
            f"错误: {result.get('error', '未知错误')}\n"
            f"{now_str}"
        )

    return _do_push(settings, "双色球数据同步", message)


def push_error_alert(
    settings: Settings, phase: str, error: str
) -> bool:
    """推送错误告警。

    Args:
        settings: 系统配置。
        phase: 失败阶段。
        error: 错误详情。
    """
    if not settings.wxpusher_token:
        return False

    today_str = date.today().strftime("%m-%d")
    message = (
        f"🔴 双色球数据同步错误 | {today_str}\n\n"
        f"❌ 阶段 [{phase}] 执行异常\n"
        f"错误: {error}\n\n"
        f"请检查日志文件"
    )

    return _do_push(settings, "双色球同步错误", message)
