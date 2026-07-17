"""双色球数据源模块：三源降级策略获取开奖数据。

数据源优先级（自动降级）：
  1. 主源 cwl.gov.cn 官方 API — JSON 接口返回结构化数据，最稳定
  2. 备源 akshare — Python 库，封装好的数据获取接口
  3. 兜底 zhcw.com — HTML 爬虫，复用现有 ssq_data_import_01.py 的逻辑

每个数据源返回统一格式的 dict 列表：
  [{"period": "2025083", "draw_date": "2025-07-17",
    "red1": 3, "red2": 9, ..., "red6": 31, "blue": 8,
    "sales_amount": ..., "pool_amount": ...}, ...]
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

from ssq_sync.logger import get_logger

logger = get_logger(__name__)

# ── 双色球红球号码排列组合的区间序号（用于爬虫解析，1-33） ──


def _try_parse_period(raw: str) -> Optional[str]:
    """尝试从原始字符串中提取标准期号（如 '2025083'）。"""
    match = re.search(r'(\d{5,7})', str(raw))
    return match.group(1) if match else None


# ════════════════════════════════════════════════════════════
#  数据源 1: cwl.gov.cn 官方 API（主源）
# ════════════════════════════════════════════════════════════

CWL_API_URL = (
    "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
)

CWL_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.cwl.gov.cn/ygkj/wqkjgg/ssq/",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}


def fetch_cwl(params: dict | None = None,
              max_pages: int = 50,
              start_date: str | None = None,
              end_date: str | None = None) -> list[dict]:
    """从 cwl.gov.cn 官方 API 获取开奖数据。

    Args:
        params: 额外的查询参数（如 pageNo, pageSize）。
        max_pages: 最大翻页数，防止无限循环。
        start_date: 起始日期 YYYY-MM-DD，用于过滤。
        end_date: 结束日期 YYYY-MM-DD，用于过滤。

    Returns:
        统一格式的开奖记录列表。
    """
    all_draws: list[dict] = []
    default_params = {
        "name": "ssq",
        "pageNo": 1,
        "pageSize": 100,
        "systemType": "PC",
    }
    if params:
        default_params.update(params)

    session = requests.Session()
    session.headers.update(CWL_HEADERS)

    for page in range(1, max_pages + 1):
        default_params["pageNo"] = page
        try:
            resp = session.get(CWL_API_URL, params=default_params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"cwl.gov.cn 第{page}页请求失败: {e}")
            break

        # 解析返回的 JSON 结构
        # 返回格式: {"result": [...], "pageCount": N, "total": N, ...}
        records = data.get("result", [])
        if not records:
            break  # 无更多数据

        for r in records:
            try:
                period = str(r.get("code", ""))
                draw_date_str = str(r.get("date", ""))
                red_str = str(r.get("red", ""))
                blue_str = str(r.get("blue", ""))

                # 红球是逗号分隔的字符串 "03,09,15,19,27,31"
                reds = [int(x) for x in red_str.split(",") if x.strip()]
                blue = int(blue_str)

                if len(reds) != 6 or not period or not draw_date_str:
                    continue

                # 日期过滤
                if start_date and draw_date_str < start_date:
                    continue
                if end_date and draw_date_str > end_date:
                    continue

                all_draws.append({
                    "period": period,
                    "draw_date": draw_date_str,
                    "red1": reds[0], "red2": reds[1], "red3": reds[2],
                    "red4": reds[3], "red5": reds[4], "red6": reds[5],
                    "blue": blue,
                    "sales_amount": _safe_float(r.get("sales")),
                    "pool_amount": _safe_float(r.get("poolmoney")),
                })
            except (ValueError, IndexError, TypeError) as e:
                logger.debug(f"跳过无效记录: {r}, 错误: {e}")
                continue

        # 检查是否还有下一页
        # 注意: cwl.gov.cn API 的 pageCount 始终为 0
        # 改用 total 和已获取记录数判断
        total = data.get("total", 0)
        if len(all_draws) >= total:
            break

        # 如果当前页过滤后没有新增有效记录，且已过 start_date，
        # 说明后续也都是更早的数据（API 返回降序），可以停止
        if start_date and page > 1:
            page_has_valid = any(
                r.get("date", "") >= start_date
                for r in records
                if r.get("date")
            )
            if not page_has_valid:
                break

        time.sleep(0.3)  # 礼貌限速

    session.close()
    if all_draws:
        logger.info(f"cwl.gov.cn: 获取 {len(all_draws)} 条记录")
    return all_draws


# ════════════════════════════════════════════════════════════
#  数据源 2: akshare（备源）
# ════════════════════════════════════════════════════════════

def fetch_akshare(start_date: str | None = None,
                  end_date: str | None = None) -> list[dict]:
    """通过 akshare 获取双色球开奖数据。

    使用 akshare.lottery_draw_detail() 获取最新一期，
    以及循环调用获取历史数据。

    Args:
        start_date: 起始日期 YYYY-MM-DD。
        end_date: 结束日期 YYYY-MM-DD。

    Returns:
        统一格式的开奖记录列表。
    """
    try:
        import akshare as ak
    except ImportError:
        logger.warning("akshare 未安装，跳过此数据源")
        return []

    draws: list[dict] = []
    try:
        # 方法1: 使用 history 接口（一次获取多期）
        # akshare 的接口名可能变化，做兼容处理
        if hasattr(ak, 'lottery_draw_detail'):
            # 获取最新一期
            try:
                df = ak.lottery_draw_detail(lottery_type="双色球")
                if df is not None and not df.empty:
                    draws = _parse_akshare_df(df, start_date, end_date)
            except Exception as e:
                logger.debug(f"akshare lottery_draw_detail 失败: {e}")

        # 方法2: 尝试 history 接口补充历史数据
        if hasattr(ak, 'lottery_history_draw_detail'):
            try:
                s = start_date or "2003-01-01"
                e = end_date or date.today().isoformat()
                df = ak.lottery_history_draw_detail(
                    lottery_type="双色球", start_date=s, end_date=e
                )
                if df is not None and not df.empty:
                    history = _parse_akshare_df(df, start_date, end_date)
                    # 合并去重
                    existing = {d["period"] for d in draws}
                    for d in history:
                        if d["period"] not in existing:
                            draws.append(d)
            except Exception as e:
                logger.debug(f"akshare lottery_history_draw_detail 失败: {e}")

        if draws:
            logger.info(f"akshare: 获取 {len(draws)} 条记录")
    except Exception as e:
        logger.warning(f"akshare 数据获取异常: {e}")

    return draws


def _parse_akshare_df(df, start_date: str | None,
                      end_date: str | None) -> list[dict]:
    """解析 akshare 返回的 DataFrame 为统一格式。"""
    draws: list[dict] = []
    for _, row in df.iterrows():
        try:
            # akshare 列名可能是中文或英文，做兼容
            row_dict = row.to_dict()

            # 尝试多种列名映射
            period = _get_col(row_dict, ["期号", "period", "code", "draw_num"])
            draw_date_str = _get_col(row_dict, ["开奖日期", "date", "draw_date", "open_date"])
            red_str = _get_col(row_dict, ["红球", "red", "red_ball", "red_balls"])
            blue_str = _get_col(row_dict, ["蓝球", "blue", "blue_ball"])

            if not period or not draw_date_str:
                continue

            # 日期格式标准化
            draw_date_str = _normalize_date(str(draw_date_str))

            # 跳过不在范围内的
            if start_date and draw_date_str < start_date:
                continue
            if end_date and draw_date_str > end_date:
                continue

            # 解析红球
            reds = _parse_numbers(red_str, count=6)
            blue = _parse_numbers(blue_str, count=1)

            if len(reds) != 6 or not blue:
                continue

            draws.append({
                "period": str(period).strip(),
                "draw_date": draw_date_str,
                "red1": reds[0], "red2": reds[1], "red3": reds[2],
                "red4": reds[3], "red5": reds[4], "red6": reds[5],
                "blue": blue[0],
            })
        except Exception:
            continue

    return draws


def _get_col(row_dict: dict, candidates: list[str]) -> Optional[str]:
    """从行字典中按候选列名查找第一个非空值。"""
    for c in candidates:
        val = row_dict.get(c)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def _parse_numbers(val, count: int) -> list[int]:
    """从字符串/列表/数字中解析号码列表。

    支持格式: "03,09,15" 或 "03 09 15" 或 [3, 9, 15]
    """
    if val is None:
        return []
    if isinstance(val, (int, float)):
        return [int(val)]
    s = str(val).strip()
    # 尝试逗号分隔
    if "," in s:
        parts = s.split(",")
    else:
        parts = s.split()
    result = []
    for p in parts:
        try:
            result.append(int(p.strip()))
        except ValueError:
            continue
    return result[:count]


def _normalize_date(s: str) -> str:
    """标准化日期格式为 YYYY-MM-DD。"""
    s = s.strip()
    # 已经是标准格式
    if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
        return s
    # YYYYMMDD 格式
    if re.match(r'^\d{8}$', s):
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    # YYYY/MM/DD 格式
    if re.match(r'^\d{4}/\d{2}/\d{2}$', s):
        return s.replace("/", "-")
    return s


def _safe_float(val) -> Optional[float]:
    """安全地解析浮点数。"""
    if val is None:
        return None
    try:
        return float(str(val).replace(",", ""))
    except (ValueError, TypeError):
        return None


# ════════════════════════════════════════════════════════════
#  数据源 3: zhcw.com 爬虫（兜底）
# ════════════════════════════════════════════════════════════

ZHCW_BASE_URL = "http://tubiao.zhcw.com/tubiao/ssqNew/ssqJsp/ssqZongHeFengBuTuAsc.jsp"

ZHCW_HEADERS = {
    "Referer": (
        "http://tubiao.zhcw.com/tubiao/ssqNew/ssqInc/"
        "ssqZongHeFengBuTuAsckj_year=2016.html"
    ),
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 6.1; WOW64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/55.0.2883.87 Safari/537.36"
    ),
}


def fetch_zhcw(year: int | None = None,
               start_date: str | None = None,
               end_date: str | None = None) -> list[dict]:
    """从 zhcw.com（中彩网）爬取开奖数据。

    这是当前项目中 ssq_data_import_01.py 使用的数据源，
    保留作为兜底方案。

    Args:
        year: 指定爬取年份，None 表示爬取当前年份。
        start_date: 过滤起始日期。
        end_date: 过滤结束日期。

    Returns:
        统一格式的开奖记录列表。
    """
    if year is None:
        year = date.today().year

    draws: list[dict] = []
    url = f"{ZHCW_BASE_URL}?kj_year={year}"

    session = requests.Session()
    session.headers.update(ZHCW_HEADERS)

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        # 处理编码
        resp.encoding = resp.apparent_encoding or "utf-8"
        html = resp.text
    except requests.RequestException as e:
        logger.warning(f"zhcw.com 请求失败: {e}")
        session.close()
        return []

    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all(class_="hgt")

    for row in rows:
        if not isinstance(row, BeautifulSoup) and not hasattr(row, "find"):
            continue

        # 期号在 class="qh7" 的 span 中
        center_elem = row.find(class_="qh7")
        if not center_elem or not center_elem.string:
            continue

        period = center_elem.string.strip()
        if period.startswith("模拟"):
            break  # 模拟数据标记，停止解析

        # 红球在 class="redqiu" 的 span 中
        red_elems = row.find_all(class_="redqiu")
        if len(red_elems) < 6:
            continue

        # 蓝球在 class="blueqiu3" 的 span 中
        blue_elem = row.find(class_="blueqiu3")
        if not blue_elem or not blue_elem.string:
            continue

        try:
            reds = [int(r.string.strip()) for r in red_elems[:6]]
            blue = int(blue_elem.string.strip())

            # 根据期号推算日期
            draw_date_str = _period_to_date(period)

            # 日期过滤
            if start_date and draw_date_str < start_date:
                continue
            if end_date and draw_date_str > end_date:
                continue

            draws.append({
                "period": period,
                "draw_date": draw_date_str,
                "red1": reds[0], "red2": reds[1], "red3": reds[2],
                "red4": reds[3], "red5": reds[4], "red6": reds[5],
                "blue": blue,
            })
        except (ValueError, AttributeError) as e:
            logger.debug(f"zhcw 解析记录失败: {period}, {e}")
            continue

    session.close()
    if draws:
        logger.info(f"zhcw.com: 获取 {len(draws)} 条记录（{year}年）")
    return draws


def _period_to_date(period: str) -> str:
    """根据期号推算大致开奖日期。

    双色球期号格式: YYYYNNN（如 2025083 = 2025年第083期）
    每期 3 天间隔（二/四/日），第一期为每年的第 1 个开奖日。

    这是一个近似推算，准确日期以官方为准。
    用于 zhcw 数据（zhcw 不返回日期字段）。
    """
    try:
        year = int(period[:4])
        num = int(period[4:])  # 第几期
    except (ValueError, IndexError):
        return f"{period[:4]}-01-01"

    # 每年约 153-156 期（每周 3 期 × 51-52 周）
    # 第一期通常在 1 月 1-3 日之间
    # 简化：用期数 × 7/3 ≈ 天数
    day_of_year = max(1, int((num - 1) * 7 / 3) + 1)

    try:
        dt = datetime(year, 1, 1) + __import__('datetime').timedelta(days=day_of_year - 1)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return f"{year}-01-01"


# ════════════════════════════════════════════════════════════
#  统一数据获取接口（三源降级）
# ════════════════════════════════════════════════════════════

def fetch_all_draws(
    start_date: str | None = None,
    end_date: str | None = None,
    include_history: bool = True,
) -> tuple[list[dict], str]:
    """三源降级获取双色球开奖数据。

    依次尝试 cwl.gov.cn → akshare → zhcw.com，
    直到成功获取到数据为止。记录使用的数据源名称。

    Args:
        start_date: 起始日期，None 表示不限制。
        end_date: 结束日期，None 表示不限制。
        include_history: 是否包含历史全量数据（zhcw 逐年的情况）。

    Returns:
        (draws_list, source_name)
        draws_list: 统一格式的开奖记录列表
        source_name: 成功获取数据的数据源名称
    """
    # ── 源 1: cwl.gov.cn 官方 API ──
    logger.info("尝试 cwl.gov.cn 官方 API...")
    draws = fetch_cwl(start_date=start_date, end_date=end_date)
    if draws:
        return draws, "cwl"

    # ── 源 2: zhcw.com 爬虫（兜底） ──
    logger.info("cwl.gov.cn 无数据，降级到 zhcw.com 爬虫...")
    all_draws: list[dict] = []
    if include_history:
        # 爬取所有历史年份（2003 至今）
        current_year = date.today().year
        for yr in range(current_year, 2002, -1):
            yr_draws = fetch_zhcw(year=yr, start_date=start_date, end_date=end_date)
            if yr_draws:
                all_draws.extend(yr_draws)
            time.sleep(0.5)  # 礼貌限速
    else:
        # 仅当前年份
        all_draws = fetch_zhcw(start_date=start_date, end_date=end_date)

    if all_draws:
        return all_draws, "zhcw"

    logger.error("所有数据源均无法获取数据！")
    return [], "none"
