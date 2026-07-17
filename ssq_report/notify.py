"""报告推送模块：通过 WxPusher 发送每日/每周/每月报告到微信。

对标 019_etf_daily_sync_and_backtest/simulation/framework/notify.py 的推送模式。
"""

from __future__ import annotations

import os
from datetime import date

from dotenv import load_dotenv
from wxpusher import WxPusher

load_dotenv()


def _get_config() -> tuple[str, list[str]]:
    """获取 WxPusher 配置。"""
    token = os.getenv("WXPUSHER_TOKEN", "")
    topic_ids_raw = os.getenv("WXPUSHER_TOPIC_IDS", '["39277"]')
    import json
    topic_ids = json.loads(topic_ids_raw)
    return token, topic_ids


def send_message(title: str, content: str, content_type: int = 1) -> bool:
    """通过 WxPusher 推送消息。

    Args:
        title: 消息标题。
        content: 消息正文。
        content_type: 1=纯文本，2=HTML。

    Returns:
        是否推送成功。
    """
    token, topic_ids = _get_config()
    if not token:
        print("[WxPusher] 未配置 Token，跳过推送")
        return False

    try:
        result = WxPusher.send_message(
            content=content,
            token=token,
            topic_ids=topic_ids,
            content_type=content_type,
        )
        if result.get("code") == 1000:
            return True
        print(f"[WxPusher] 推送失败: {result}")
        return False
    except Exception as e:
        print(f"[WxPusher] 推送异常: {e}")
        return False


def push_daily_report(report_text: str) -> bool:
    """推送每日预测报告。

    Args:
        report_text: 报告文本（由 reporter.py 生成）。

    Returns:
        是否推送成功。
    """
    today = date.today().strftime("%Y-%m-%d")
    title = f"🔴 双色球日报 | {today}"
    return send_message(title, report_text)


def push_simple_notice(content: str) -> bool:
    """推送简单通知（非交易日、跳过等）。"""
    today = date.today().strftime("%Y-%m-%d")
    return send_message(f"🔴 双色球 | {today}", content)


def push_error_alert(phase: str, error: str) -> bool:
    """推送错误告警。"""
    today = date.today().strftime("%m-%d")
    content = f"❌ 双色球管线异常 | {today}\n\n阶段 [{phase}] 执行失败\n错误: {error}"
    return send_message(f"❌ 双色球管线异常", content)
