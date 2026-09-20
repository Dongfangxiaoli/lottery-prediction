"""
双色球历史开奖数据获取器
数据源: 500.com HTML 页面 (CWL API 已于 2026 年失效)
"""
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

# 绕过本地代理
for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(key, None)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "ssq_history.csv")

HISTORY_URL = "https://datachart.500.com/ssq/history/newinc/history.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Referer": "https://datachart.500.com/ssq/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """按开奖日期排序并合并同一期的重复抓取记录。"""
    if df.empty:
        return df
    clean = df.copy()
    clean["issue"] = clean["issue"].astype(str)
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean = clean.dropna(subset=["date"])
    # 同一日期若同时存在 2026036/26036 两种期号，保留信息更完整的长格式记录。
    clean["_issue_len"] = clean["issue"].str.len()
    clean = clean.sort_values(["date", "_issue_len"], ascending=[True, False], kind="stable")
    clean = clean.drop_duplicates(subset="date", keep="first").drop(columns="_issue_len")
    return clean.sort_values("date", kind="stable").reset_index(drop=True)


def _parse_500_html(html: str) -> pd.DataFrame:
    """解析 500.com 双色球历史数据 HTML 表格"""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        rows = soup.find_all("tr")
    else:
        rows = table.find_all("tr")

    data_rows = []
    for tr in rows:
        cells = tr.find_all("td")
        # 500.com SSQ 表格列：期号, 红1~6, 蓝, 销售额, 奖池, 日期...
        if len(cells) < 9:
            continue
        try:
            issue = cells[0].get_text(strip=True)
            if not issue or not issue.isdigit():
                continue

            reds = [int(cells[j].get_text(strip=True)) for j in range(1, 7)]
            blue = int(cells[7].get_text(strip=True))

            # 日期通常在第14或15列
            date_str = ""
            for j in range(len(cells) - 1, 7, -1):
                t = cells[j].get_text(strip=True)
                if "-" in t and len(t) >= 8:
                    date_str = t
                    break

            def _amount(index: int) -> int:
                try:
                    return int(cells[index].get_text(strip=True).replace(",", "").replace("元", ""))
                except (ValueError, IndexError):
                    return 0

            # 当前 500.com 表格：奖池在第10列，销量紧邻末尾日期列。
            # 不能按“第一个大整数”猜列，否则两者会互换。
            date_idx = next(
                (j for j in range(len(cells) - 1, 7, -1)
                 if "-" in cells[j].get_text(strip=True)),
                -1,
            )
            pool = _amount(9)
            sales = _amount(date_idx - 1) if date_idx > 0 else 0

            data_rows.append({
                "issue": issue,
                "date": date_str,
                "red1": reds[0], "red2": reds[1], "red3": reds[2],
                "red4": reds[3], "red5": reds[4], "red6": reds[5],
                "blue": blue,
                "sales": sales,
                "pool": pool,
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(data_rows)
    if not df.empty:
        df = _normalize_history(df)
    return df


def fetch_all(progress_callback=None) -> pd.DataFrame:
    """一次性拉取 500.com 全部双色球历史数据"""
    if progress_callback:
        progress_callback(0.2, "正在从 500.com 获取双色球全部历史数据...")

    try:
        resp = requests.get(
            HISTORY_URL,
            params={"limit": 9999, "sort": 0},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "gb2312"

        if progress_callback:
            progress_callback(0.6, "解析 HTML 表格...")

        df = _parse_500_html(resp.text)

        if progress_callback:
            progress_callback(1.0, f"获取完成，共 {len(df)} 期。")

        return df

    except Exception as e:
        print(f"[ERROR] 双色球全量获取失败: {e}")
        return pd.DataFrame()


def fetch_incremental(progress_callback=None) -> pd.DataFrame:
    """增量更新：读取本地 CSV，拉取 500.com 最新数据追加。无本地则全量。"""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 10:
        raw_existing = pd.read_csv(CSV_PATH, dtype={"issue": str})
        existing = _normalize_history(raw_existing)
        raw_dates = pd.to_datetime(raw_existing["date"], errors="coerce")
        repaired = len(existing) != len(raw_existing) or not raw_dates.is_monotonic_increasing
        last_issue = existing.iloc[-1]["issue"]
        if progress_callback:
            progress_callback(0.3, f"本地已有 {len(existing)} 期，最新 {last_issue}，检查更新...")

        # 历史缺口只需全量回填一次；之后每次只拉最近 100 期。
        request_limit = 9999 if existing["date"].min() > pd.Timestamp("2003-03-01") else 100
        # 快速尝试在线更新（5秒超时，不卡主流程）
        try:
            resp = requests.get(
                HISTORY_URL,
                params={"limit": request_limit, "sort": 0},
                headers=HEADERS,
                timeout=5,
            )
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "gb2312"
            new_df = _parse_500_html(resp.text)
        except Exception:
            new_df = pd.DataFrame()

        # 同期同格式时优先采用刚抓取的数据，可修复源列解析升级前留下的派生字段。
        combined = _normalize_history(pd.concat([new_df, existing], ignore_index=True))
        added = max(0, len(combined) - len(existing))
        corrected = not combined.equals(existing)
        if added == 0 and not repaired and not corrected:
            if progress_callback:
                progress_callback(1.0, "已是最新数据，无需更新。")
            return existing

        combined.to_csv(CSV_PATH, index=False)
        if progress_callback:
            repair_msg = f"，清理重复 {len(raw_existing) - len(existing)} 期" if repaired else ""
            correction_msg = "，校正已有记录" if corrected else ""
            progress_callback(1.0, f"新增 {added} 期{repair_msg}{correction_msg}，共 {len(combined)} 期。")
        return combined
    else:
        if os.path.exists(CSV_PATH):
            os.remove(CSV_PATH)
        df = fetch_all(progress_callback)
        if not df.empty:
            df.to_csv(CSV_PATH, index=False)
        if progress_callback:
            progress_callback(1.0, f"全量获取完成，共 {len(df)} 期。")
        return df


if __name__ == "__main__":
    print("双色球数据获取器 (500.com)")
    df = fetch_incremental(lambda p, m: print(f"  [{p:.0%}] {m}"))
    print(f"共获取 {len(df)} 期数据")
    if not df.empty:
        print(df.tail())
