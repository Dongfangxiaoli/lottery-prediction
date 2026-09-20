"""排列3、排列5、7星彩的 500.com 历史开奖抓取器。"""
import os

import pandas as pd
import requests
from bs4 import BeautifulSoup

from game_config import DIGIT_GAME_CONFIGS, digit_columns, game_code, game_name

for _proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_proxy_key, None)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _csv_path(code: str) -> str:
    return os.path.join(DATA_DIR, f"{code}_history.csv")


def _empty(code: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["issue", "date", *digit_columns(code), "sales"])


def _valid_digits(code: str, digits: list[int]) -> bool:
    classes = DIGIT_GAME_CONFIGS[code]["classes"]
    return len(digits) == len(classes) and all(0 <= digit < limit for digit, limit in zip(digits, classes))


def _normalize_history(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """只按期号去重，并按日期、期号稳定排序；不按日期丢弃记录。"""
    if df.empty:
        return _empty(code)
    columns = ["issue", "date", *digit_columns(code), "sales"]
    clean = df.reindex(columns=columns).copy()
    clean["issue"] = clean["issue"].astype(str).str.strip()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean = clean[clean["issue"].str.fullmatch(r"\d+") & clean["date"].notna()]
    if code == "qxc":
        clean = clean[clean["issue"].astype(int) >= DIGIT_GAME_CONFIGS[code]["min_issue"]]
    for column, limit in zip(digit_columns(code), DIGIT_GAME_CONFIGS[code]["classes"]):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean = clean[clean[column].between(0, limit - 1)]
        clean[column] = clean[column].astype(int)
    clean["sales"] = pd.to_numeric(clean["sales"], errors="coerce").fillna(0).astype(int)
    # 在线抓取的数据位于 concat 前部，因此同一期优先采用它；mergesort 保证稳定。
    clean = clean.drop_duplicates(subset="issue", keep="first")
    clean["_issue_number"] = clean["issue"].astype(int)
    clean = clean.sort_values(["date", "_issue_number"], kind="mergesort").drop(columns="_issue_number")
    clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")
    return clean.reset_index(drop=True)


def parse(html: str, value: str) -> pd.DataFrame:
    """解析一个 500.com 历史表；坏行及不符合现行 7星彩规则的旧期会被跳过。"""
    code = game_code(value)
    config = DIGIT_GAME_CONFIGS[code]
    table = BeautifulSoup(html, "html.parser").find("table", id="tablelist")
    if table is None:
        return _empty(code)
    records = []
    for row in table.select("tr.t_tr1"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        issue = cells[0].get_text(" ", strip=True)
        numbers = cells[1].get_text(" ", strip=True).split()
        try:
            digits = [int(number) for number in numbers]
        except ValueError:
            continue
        if not issue.isdigit() or not _valid_digits(code, digits):
            continue
        date = next((cell.get_text(strip=True) for cell in reversed(cells)
                     if cell.get_text(strip=True).count("-") == 2), "")
        if not date:
            continue
        sales = 0
        # 销售额位于号码后、日期前：取其中第一个可解析整数，避免把“和值”当销售额。
        for cell in cells[2:-1]:
            text = cell.get_text(strip=True).replace(",", "").replace("元", "")
            if text.isdigit() and len(text) >= 5:
                sales = int(text)
                break
        records.append({"issue": issue, "date": date, **dict(zip(config["digits"], digits)), "sales": sales})
    return _normalize_history(pd.DataFrame(records), code)


def _request(code: str, limit: int, timeout: int) -> pd.DataFrame:
    if not isinstance(limit, int) or not 1 <= limit <= 99999:
        raise ValueError("limit 必须是 1 到 99999 的整数")
    response = requests.get(
        DIGIT_GAME_CONFIGS[code]["history_url"],
        params={"limit": limit, "start": "00001", "end": "99999"},
        headers={**HEADERS, "Referer": f"https://datachart.500.com/{code}/"},
        timeout=timeout,
    )
    response.raise_for_status()
    response.encoding = "gb2312"
    return parse(response.text, code)


def fetch_all(value: str, progress_callback=None) -> pd.DataFrame:
    """获取指定数字彩种的全部可用历史数据（7星彩自动排除旧规则期）。"""
    code = game_code(value)
    if progress_callback:
        progress_callback(0.2, f"正在从 500.com 获取{game_name(code)}全部历史数据...")
    try:
        data = _request(code, 99999, 30)
    except Exception as exc:
        print(f"[ERROR] {game_name(code)}全量获取失败: {exc}")
        return _empty(code)
    if progress_callback:
        progress_callback(1.0, f"获取完成，共 {len(data)} 期。")
    return data


def fetch_incremental(value: str, progress_callback=None) -> pd.DataFrame:
    """按期号合并最近记录并持久化；同日多期也会完整保留。"""
    code = game_code(value)
    path = _csv_path(code)
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) <= 10:
        data = fetch_all(code, progress_callback)
        if not data.empty:
            data.to_csv(path, index=False)
        return data
    try:
        existing = _normalize_history(pd.read_csv(path, dtype={"issue": str}), code)
    except Exception:
        existing = _empty(code)
    if progress_callback:
        latest = existing.iloc[-1]["issue"] if not existing.empty else "无"
        progress_callback(0.3, f"本地已有 {len(existing)} 期，最新 {latest}，检查更新...")
    try:
        online = _request(code, 100, 10)
    except Exception as exc:
        print(f"[ERROR] {game_name(code)}增量获取失败: {exc}")
        online = _empty(code)
    combined = _normalize_history(pd.concat([online, existing], ignore_index=True), code)
    added = len(set(combined["issue"]) - set(existing["issue"]))
    if not combined.equals(existing):
        combined.to_csv(path, index=False)
    if progress_callback:
        progress_callback(1.0, f"新增 {added} 期，共 {len(combined)} 期。")
    return combined


if __name__ == "__main__":
    for _code in DIGIT_GAME_CONFIGS:
        _data = fetch_incremental(_code)
        print(f"{game_name(_code)}: {len(_data)} 期")
