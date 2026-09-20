"""
预测回顾模块：对比上一期预测与最新开奖结果，按命中数降序排列。
"""
import os
import json
import numpy as np
import pandas as pd

from game_config import DIGIT_GAME_CONFIGS, GAME_NAMES, game_code, is_digit_game, normalize_play

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "predictions")


def _valid_ticket(game: str, item: dict, play: str) -> bool:
    """读取历史文件时校验票面，避免非法数字和组选形态进入统计。"""
    if not isinstance(item, dict):
        return False
    try:
        if is_digit_game(game):
            values = _prediction_digits(item)
            limits = DIGIT_GAME_CONFIGS[game]["classes"]
            if len(values) != len(limits) or any(not 0 <= v < n for v, n in zip(values, limits)):
                return False
            if game == "pls" and play != "直选":
                return len(set(values)) == (2 if play == "组选3" else 3)
            return True
        zones = ((item["red"], 6, 33), ([item["blue"]], 1, 16)) if game == "ssq" else (
            (item["front"], 5, 35), (item["back"], 2, 12))
        return all(isinstance(values, list) and len(values) == count
                   and len(set(values)) == count
                   and all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= limit for v in values)
                   for values, count, limit in zones)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _read_prediction_records(game: str, play: str = None) -> tuple[list[dict], int]:
    """各历史页面共用读取入口；保留坏文件原件并报告被排除的数量。"""
    g = game_code(game)
    wanted = normalize_play(g, play)
    if not os.path.isdir(RESULTS_DIR):
        return [], 0
    records, skipped = [], 0
    for filename in sorted(os.listdir(RESULTS_DIR)):
        if not filename.startswith(g + "_") or not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(RESULTS_DIR, filename), encoding="utf-8") as fp:
                record = json.load(fp)
            if not isinstance(record, dict) or record.get("game") != g:
                raise ValueError("彩种不匹配")
            record_play = normalize_play(g, record.get("play"))
            if record_play != wanted:
                continue
            numbers = record.get("numbers")
            if (not isinstance(record.get("timestamp"), str) or not record["timestamp"]
                    or not isinstance(numbers, list) or not numbers
                    or not all(_valid_ticket(g, n, record_play) for n in numbers)
                    or any(not np.isfinite(float(n.get("prob", 0))) for n in numbers)):
                raise ValueError("预测记录字段不合法")
            records.append(record)
        except (OSError, UnicodeError, ValueError, TypeError, OverflowError):
            skipped += 1
    records.sort(key=lambda r: r["timestamp"])
    return records, skipped


def load_prediction_history(game: str, play: str = None) -> str:
    from predictor import format_numbers_copy
    g = game_code(game)
    effective_play = normalize_play(g, play)
    records, skipped = _read_prediction_records(g, effective_play)
    lines = [f"已跳过 {skipped} 个损坏或不合法的预测文件，原文件保留。"] if skipped else []
    for record in reversed(records[-20:]):
        lines.append(f"[{record['timestamp']}] {GAME_NAMES[g]}（{effective_play}）")
        strategy = record.get("selection_strategy", "weighted")
        lines.append("算法：最高奖联合排序5注（实验）" if strategy == "joint_top5" else "算法：随机加权（旧/对照）")
        lines.append(format_numbers_copy(g, record["numbers"]))
        lines.append("")
    if not records:
        lines.append(f"暂无 {GAME_NAMES[g]}（{effective_play}）预测记录。")
    return "\n".join(lines)


def _record_play(game: str, record: dict) -> str:
    """旧数字彩 JSON 缺少 play 时按历史直选解释。"""
    return normalize_play(game, record.get("play")) if is_digit_game(game) else ""


def _load_latest_prediction(game: str, play: str = None) -> dict | None:
    records, _ = _read_prediction_records(game, play)
    return records[-1] if records else None


def _load_latest_draw(game: str, prediction: dict) -> dict | None:
    """获取该预测数据截止期之后的下一期开奖。"""
    g = game_code(game)

    csv_map = {
        "ssq": os.path.join(os.path.dirname(__file__), "data", "ssq_history.csv"),
        "dlt": os.path.join(os.path.dirname(__file__), "data", "dlt_history.csv"),
    }
    csv_map.update({g: os.path.join(os.path.dirname(__file__), "data", f"{g}_history.csv")
                    for g in DIGIT_GAME_CONFIGS})

    csv_path = csv_map[g]
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path, dtype={"issue": str})
    if df.empty:
        return None

    return _find_draw_after(g, prediction, df)


def review_prediction(game: str, play: str = None):
    """主入口：对比最新预测与最新开奖，返回 (文本摘要, DataFrame)"""
    g = game_code(game)
    game_name = GAME_NAMES.get(g, g)

    effective_play = normalize_play(g, play) if is_digit_game(g) else None
    records, skipped = _read_prediction_records(g, effective_play)
    pred = records[-1] if records else None
    notice = f"已跳过 {skipped} 个损坏或不合法的预测文件，原文件保留。\n" if skipped else ""

    if pred is None:
        return notice + f"❌ 暂无 {game_name} 预测记录。", None
    if not pred.get("source_date"):
        return notice + f"⚠️ 最新 {game_name} 记录缺少数据截止期，无法可靠匹配开奖；请重新生成预测。", None

    draw = _load_latest_draw(g, pred)
    if draw is None:
        return notice + f"⏳ {game_name} 数据中尚无该预测之后的下一期开奖，或数据截止期无效，请先更新数据。", None

    if g == "ssq":
        text, table = _review_ssq(pred, draw)
    elif g == "dlt":
        text, table = _review_dlt(pred, draw)
    else:
        text, table = _review_numeric(pred, draw, g, effective_play)
    return notice + text, table


def _prediction_digits(item: dict) -> list[int]:
    """兼容数字型预测记录的常见字段，始终保留 0。"""
    values = item.get("digits", item.get("number", item.get("nums", item.get("values"))))
    if isinstance(values, str):
        cleaned = values.replace(",", " ").replace("|", " ").strip()
        values = list(cleaned) if cleaned.isdigit() else cleaned.split()
        if len(values) == 2 and len(values[0]) == 6 and values[0].isdigit():
            values = list(values[0]) + values[1:]
    if values is None:
        return []
    if any(isinstance(x, bool) or str(x).strip() not in {str(n) for n in range(15)} for x in values):
        raise ValueError("号码必须为 0–14 范围内的整数")
    return [int(x) for x in values]


def _row_digits(row, game: str) -> list[int]:
    raw = row.get("digits") if hasattr(row, "get") else None
    if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
        return [int(x) for x in str(raw).replace(",", " ").split()]
    cfg = DIGIT_GAME_CONFIGS.get(game, {})
    columns = cfg.get("columns") or cfg.get("digits") or [f"digit{i}" for i in range(1, int(cfg.get("width", 0)) + 1)]
    return [int(row[c]) for c in columns]


def _review_numeric(pred: dict, draw: dict, game: str, play: str = None) -> tuple:
    actual = draw["digits"]
    width = len(actual)
    effective_play = normalize_play(game, play)
    grouped_pls = game == "pls" and effective_play != "直选"
    lines = [f"## 最新开奖：第 {draw['issue']} 期（{draw['date']}）",
             f"**开奖号码：{' '.join(str(x) for x in actual)}**",
             f"玩法：{effective_play}　|　预测时间：{pred.get('timestamp', '')}　|　共 {len(pred.get('numbers', []))} 组预测", ""]
    results = []
    for i, item in enumerate(pred.get("numbers", [])):
        values = _prediction_digits(item)
        if len(values) != width:
            continue
        hits = [a == b for a, b in zip(values, actual)]
        group_hit = np.array_equal(np.sort(values), np.sort(actual)) if grouped_pls else all(hits)
        row = {"原组号": i + 1, "预测号码": " ".join(map(str, values)), "整组命中": "✓" if group_hit else "✗"}
        if not grouped_pls:
            row["预测号码"] = " ".join(f"{'【' if h else ''}{v}{'】' if h else ''}" for v, h in zip(values, hits))
            row["位置命中"] = sum(hits)
        results.append(row)
    if not results:
        return "⚠️ 没有可回顾的合法数字型预测记录。", None
    results.sort(key=lambda x: (x["整组命中"] == "✓", -x["原组号"]), reverse=True) if grouped_pls else results.sort(key=lambda x: (x["位置命中"], x["整组命中"] == "✓", -x["原组号"]), reverse=True)
    lines += [("### 按整注命中排列：" if grouped_pls else "### 按位置命中数降序排列："), ""]
    for rank, row in enumerate(results, 1):
        detail = f"整组命中 {row['整组命中']}" if grouped_pls else f"位置命中 **{row['位置命中']}/{width}**　整组命中 {row['整组命中']}"
        lines.append(f"**第{rank}名**（原第{row['原组号']}组）：{detail}　→　{row['预测号码']}")
    base = {"组选3": 0.003, "组选6": 0.006}.get(effective_play)
    if grouped_pls:
        lines += ["", f"整组命中：{sum(x['整组命中'] == '✓' for x in results)}/{len(results)}　|　理论单注概率：{base:.3%}", "组选按号码多重集判定，不统计位置命中。"]
    else:
        lines += ["", f"最佳位置命中：**{max(x['位置命中'] for x in results)}/{width}**　|　整组命中：{sum(x['整组命中'] == '✓' for x in results)}/{len(results)}", "【】标记为命中位置"]
    df = pd.DataFrame(results); df.insert(0, "排名", range(1, len(df) + 1))
    return "\n".join(lines), df


def _review_ssq(pred: dict, draw: dict) -> tuple:
    """双色球预测回顾"""
    draw_reds = set(draw["reds"])
    draw_blue = draw["blue"]

    text_lines = [
        f"## 最新开奖：第 {draw['issue']} 期（{draw['date']}）",
        f"**红球：{'  '.join(f'{r:02d}' for r in draw['reds'])}　蓝球：{draw_blue:02d}**",
        f"预测时间：{pred['timestamp']}　|　共 {len(pred['numbers'])} 组预测",
        "",
    ]

    results = []
    for i, n in enumerate(pred["numbers"]):
        pred_reds = set(n["red"])
        pred_blue = n["blue"]

        matched_reds = sorted(pred_reds & draw_reds)
        matched_blue = pred_blue == draw_blue
        total_hits = len(matched_reds) + (1 if matched_blue else 0)

        # 标记命中红球
        red_display_parts = []
        for r in n["red"]:
            if r in draw_reds:
                red_display_parts.append(f"【{r:02d}】")
            else:
                red_display_parts.append(f"{r:02d}")

        red_display = " ".join(red_display_parts)

        results.append({
            "原组号": i + 1,
            "红球号码": red_display,
            "命中红球": len(matched_reds),
            "蓝球": f"{'✓ ' if matched_blue else ''}{n['blue']:02d}",
            "蓝球命中": "✓" if matched_blue else "✗",
            "总命中": total_hits,
        })

    results.sort(key=lambda x: (x["总命中"], x["命中红球"], -x["原组号"]), reverse=True)

    text_lines.append("### 按命中数降序排列：")
    text_lines.append("")
    for rank, r in enumerate(results, 1):
        blue_tag = " ✓蓝球命中" if r["蓝球命中"] == "✓" else ""
        red_info = f"红球中{r['命中红球']}个" if r["命中红球"] > 0 else "红球未中"
        text_lines.append(
            f"**第{rank}名**（原第{r['原组号']}组）：{red_info}{blue_tag}　→　总命中 **{r['总命中']}**"
        )
        text_lines.append(f"　{r['红球号码']}　|　蓝{r['蓝球'][-3:].strip()}")
        text_lines.append("")

    # 汇总统计
    max_hit = max(r["总命中"] for r in results)
    blue_hit_count = sum(1 for r in results if r["蓝球命中"] == "✓")
    text_lines.append("---")
    text_lines.append(f"最佳命中：**{max_hit}**　|　蓝球命中组数：**{blue_hit_count}/{len(results)}**")
    text_lines.append(f"【】标记的号码为命中号码")

    df = pd.DataFrame(results)
    df.insert(0, "排名", range(1, len(results) + 1))

    return "\n".join(text_lines), df


def _review_dlt(pred: dict, draw: dict) -> tuple:
    """大乐透预测回顾"""
    draw_fronts = set(draw["fronts"])
    draw_backs = set(draw["backs"])

    text_lines = [
        f"## 最新开奖：第 {draw['issue']} 期（{draw['date']}）",
        f"**前区：{'  '.join(f'{r:02d}' for r in draw['fronts'])}　后区：{'  '.join(f'{r:02d}' for r in draw['backs'])}**",
        f"预测时间：{pred['timestamp']}　|　共 {len(pred['numbers'])} 组预测",
        "",
    ]

    results = []
    for i, n in enumerate(pred["numbers"]):
        pred_fronts = set(n["front"])
        pred_backs = set(n["back"])

        matched_fronts = sorted(pred_fronts & draw_fronts)
        matched_backs = sorted(pred_backs & draw_backs)
        total_hits = len(matched_fronts) + len(matched_backs)

        front_parts = []
        for x in n["front"]:
            if x in draw_fronts:
                front_parts.append(f"【{x:02d}】")
            else:
                front_parts.append(f"{x:02d}")

        back_parts = []
        for x in n["back"]:
            if x in draw_backs:
                back_parts.append(f"【{x:02d}】")
            else:
                back_parts.append(f"{x:02d}")

        results.append({
            "原组号": i + 1,
            "前区号码": " ".join(front_parts),
            "命中前区": len(matched_fronts),
            "后区号码": " ".join(back_parts),
            "命中后区": len(matched_backs),
            "总命中": total_hits,
        })

    results.sort(key=lambda x: (x["总命中"], x["命中前区"], x["命中后区"], -x["原组号"]), reverse=True)

    text_lines.append("### 按命中数降序排列：")
    text_lines.append("")
    for rank, r in enumerate(results, 1):
        front_info = f"前区中{r['命中前区']}个" if r["命中前区"] > 0 else "前区未中"
        back_info = f"后区中{r['命中后区']}个" if r["命中后区"] > 0 else "后区未中"
        text_lines.append(
            f"**第{rank}名**（原第{r['原组号']}组）：{front_info}，{back_info}　→　总命中 **{r['总命中']}**"
        )
        text_lines.append(f"　{r['前区号码']}　|　{r['后区号码']}")
        text_lines.append("")

    max_hit = max(r["总命中"] for r in results)
    text_lines.append("---")
    text_lines.append(f"最佳命中：**{max_hit}**")
    text_lines.append(f"【】标记的号码为命中号码")

    df = pd.DataFrame(results)
    df.insert(0, "排名", range(1, len(results) + 1))

    return "\n".join(text_lines), df


# ====== 新增：长期统计 + 理论vs实际 + 置信度校准 ======

def _load_all_predictions(game: str, play: str = None) -> list[dict]:
    """加载某彩种所有预测记录，按时间升序返回"""
    records, _ = _read_prediction_records(game, play)
    return records


def _find_draw_after(game: str, prediction: dict, df: pd.DataFrame) -> dict | None:
    """按预测保存的数据截止期号/日期，找到严格晚于它的第一期开奖。"""
    try:
        source_date = pd.to_datetime(prediction["source_date"])
    except Exception:
        return None
    if pd.isna(source_date):
        return None
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"])
    df2["_issue_number"] = pd.to_numeric(df2["issue"], errors="coerce")
    df2 = df2.sort_values(["date", "_issue_number"], kind="mergesort")
    source_issue = pd.to_numeric(prediction.get("source_issue"), errors="coerce")
    if pd.notna(source_issue):
        future = df2[
            (df2["date"] > source_date)
            | ((df2["date"] == source_date) & (df2["_issue_number"] > source_issue))
        ]
    else:
        future = df2[df2["date"] > source_date]
    if future.empty:
        return None
    latest = future.iloc[0]
    g = game_code(game)
    if g == "ssq":
        return {
            "issue": str(latest["issue"]),
            "date": str(latest["date"])[:10],
            "reds": [int(latest[f"red{i}"]) for i in range(1, 7)],
            "blue": int(latest["blue"]),
        }
    if g == "dlt":
        return {
            "issue": str(latest["issue"]),
            "date": str(latest["date"])[:10],
            "fronts": [int(latest[f"front{i}"]) for i in range(1, 6)],
            "backs": [int(latest[f"back{i}"]) for i in range(1, 3)],
        }
    return {"issue": str(latest["issue"]), "date": str(latest["date"])[:10],
            "digits": _row_digits(latest, g)}


def _random_expected_ssq() -> dict:
    """双色球单注随机命中期望：红球 C(6,k)C(27,6-k)/C(33,6)，蓝球 1/16"""
    from math import comb
    total = comb(33, 6)
    red_expected = {k: comb(6, k) * comb(27, 6 - k) / total for k in range(7)}
    mean_red = sum(k * p for k, p in red_expected.items())
    return {"red_dist": red_expected, "mean_red": mean_red, "blue_rate": 1 / 16}


def _random_expected_dlt() -> dict:
    """大乐透单注随机命中期望"""
    from math import comb
    f_total = comb(35, 5)
    f_dist = {k: comb(5, k) * comb(30, 5 - k) / f_total for k in range(6)}
    mean_front = sum(k * p for k, p in f_dist.items())
    b_total = comb(12, 2)
    b_dist = {k: comb(2, k) * comb(10, 2 - k) / b_total for k in range(3)}
    mean_back = sum(k * p for k, p in b_dist.items())
    return {"front_dist": f_dist, "mean_front": mean_front,
            "back_dist": b_dist, "mean_back": mean_back}


def _random_expected_digits(game: str, play: str = None) -> dict:
    cfg = DIGIT_GAME_CONFIGS.get(game, {})
    domains = list(cfg.get("classes", ())) or [10] * int(cfg.get("width", 0))
    effective_play = normalize_play(game, play)
    group_rate = ({"直选": .001, "组选3": .003, "组选6": .006}[effective_play]
                  if game == "pls" else float(np.prod([1 / n for n in domains])))
    return {"mean_position": sum(1 / n for n in domains), "group_rate": group_rate}


def review_history_stats(game: str, play: str = None) -> tuple[str, "pd.DataFrame", object]:
    """
    按实际注数汇总历史命中，对比理论随机期望；相对倾向分仅用于观察。
    返回: (文本摘要, 逐记录明细表, 对照图 matplotlib figure)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = game_code(game)
    game_name = GAME_NAMES.get(g, g)

    effective_play = normalize_play(g, play) if is_digit_game(g) else None
    records, skipped = _read_prediction_records(g, effective_play)
    notice = f"已跳过 {skipped} 个损坏或不合法的预测文件，原文件保留。\n" if skipped else ""
    if not records:
        return notice + f"❌ 暂无 {game_name} 预测记录。", None, None

    csv_path = os.path.join(os.path.dirname(__file__), "data", f"{g}_history.csv")
    if not os.path.exists(csv_path):
        return f"❌ 缺少 {game_name} 开奖数据。", None, None
    df = pd.read_csv(csv_path, dtype={"issue": str})

    rows = []
    calib_points = []  # list of (prob, hit_count)
    unverifiable = 0

    for rec in records:
        ts = rec.get("timestamp", "")
        draw = _find_draw_after(g, rec, df)
        if draw is None:
            unverifiable += 1
            continue
        numbers = rec.get("numbers", [])
        if not numbers:
            continue

        if g == "ssq":
            draw_reds = set(draw["reds"])
            draw_blue = draw["blue"]
            best_red = 0
            blue_hit_any = False
            for n in numbers:
                rh = len(set(n["red"]) & draw_reds)
                bh = n["blue"] == draw_blue
                if rh > best_red:
                    best_red = rh
                if bh:
                    blue_hit_any = True
                calib_points.append((n.get("prob", 0), rh + (1 if bh else 0)))
            avg_red = np.mean([len(set(n["red"]) & draw_reds) for n in numbers])
            blue_rate = np.mean([1 if n["blue"] == draw_blue else 0 for n in numbers])
            rows.append({
                "预测时间": ts,
                "对应期号": draw["issue"],
                "组数": len(numbers),
                "平均红球命中": float(avg_red),
                "最佳红球命中": best_red,
                "蓝球命中率": float(blue_rate),
            })
        elif g == "dlt":
            draw_fronts = set(draw["fronts"])
            draw_backs = set(draw["backs"])
            best_f = 0
            best_b = 0
            for n in numbers:
                fh = len(set(n["front"]) & draw_fronts)
                bh = len(set(n["back"]) & draw_backs)
                if fh > best_f:
                    best_f = fh
                if bh > best_b:
                    best_b = bh
                calib_points.append((n.get("prob", 0), fh + bh))
            avg_f = np.mean([len(set(n["front"]) & draw_fronts) for n in numbers])
            avg_b = np.mean([len(set(n["back"]) & draw_backs) for n in numbers])
            rows.append({
                "预测时间": ts,
                "对应期号": draw["issue"],
                "组数": len(numbers),
                "平均前区命中": float(avg_f),
                "最佳前区命中": best_f,
                "平均后区命中": float(avg_b),
            })
        else:
            actual = draw["digits"]
            valid = [(n, _prediction_digits(n)) for n in numbers]
            valid = [(n, d) for n, d in valid if len(d) == len(actual)]
            if not valid:
                continue
            grouped_pls = g == "pls" and effective_play != "直选"
            position_hits = [sum(a == b for a, b in zip(d, actual)) for _, d in valid]
            group_hits = [int(np.array_equal(np.sort(d), np.sort(actual))) if grouped_pls
                          else int(all(a == b for a, b in zip(d, actual))) for _, d in valid]
            for (n, _), hits in zip(valid, group_hits if grouped_pls else position_hits):
                calib_points.append((n.get("prob", 0), hits))
            row = {"预测时间": ts, "对应期号": draw["issue"], "组数": len(valid),
                   "整组命中率": float(np.mean(group_hits))}
            if not grouped_pls:
                row.update({"平均位置命中": float(np.mean(position_hits)),
                            "最佳位置命中": max(position_hits)})
            rows.append(row)

    if not rows:
        return (
            notice + f"⚠️ {game_name} 现有 {len(records)} 条记录均缺少可验证的数据截止期，"
            "或下一期开奖尚未入库；旧记录不再猜测匹配。",
            None,
            None,
        )

    detail_df = pd.DataFrame(rows)
    n_records = len(rows)
    weights = detail_df["组数"].to_numpy()

    if g == "ssq":
        theo = _random_expected_ssq()
        mean_red_actual = float(np.average(detail_df["平均红球命中"], weights=weights))
        mean_red_theo = theo["mean_red"]
        blue_rate_actual = float(np.average(detail_df["蓝球命中率"], weights=weights))
        blue_rate_theo = theo["blue_rate"]
    elif g == "dlt":
        theo = _random_expected_dlt()
        mean_f_actual = float(np.average(detail_df["平均前区命中"], weights=weights))
        mean_f_theo = theo["mean_front"]
        mean_b_actual = float(np.average(detail_df["平均后区命中"], weights=weights))
        mean_b_theo = theo["mean_back"]
    else:
        theo = _random_expected_digits(g, effective_play)
        grouped_pls = g == "pls" and effective_play != "直选"
        if not grouped_pls:
            mean_pos_actual = float(np.average(detail_df["平均位置命中"], weights=weights))
            mean_pos_theo = theo["mean_position"]
        group_rate_actual = float(np.average(detail_df["整组命中率"], weights=weights))
        group_rate_theo = theo["group_rate"]

    lines = []
    lines.append("=" * 60)
    lines.append(f"  {game_name} 预测长期命中统计（{n_records} 条有效记录）")
    lines.append("=" * 60)
    if notice:
        lines.append(notice.strip())
    lines.append(f"按实际注数加权：共 {int(weights.sum())} 注，匹配 {detail_df['对应期号'].nunique()} 个开奖期。")
    lines.append("同一期可包含多次选号；本页为历史记录描述统计，不能据此判定预测优势。")
    if unverifiable:
        lines.append(f"已排除 {unverifiable} 条无法可靠匹配开奖的旧/未开奖记录。")
    if g == "ssq":
        lines.append(f"【红球】实际平均命中 {mean_red_actual:.3f}  vs  理论随机 {mean_red_theo:.3f}")
        diff = mean_red_actual - mean_red_theo
        lines.append(f"        样本差值 {diff:+.3f}（未作显著性判断）")
        lines.append(f"【蓝球】实际命中率 {blue_rate_actual:.2%}  vs  理论 {blue_rate_theo:.2%}")
        lines.append(f"【最佳】最高单期红球命中: {detail_df['最佳红球命中'].max()}")
    elif g == "dlt":
        lines.append(f"【前区】实际平均 {mean_f_actual:.3f}  vs  理论随机 {mean_f_theo:.3f}")
        diff = mean_f_actual - mean_f_theo
        lines.append(f"        样本差值 {diff:+.3f}（未作显著性判断）")
        lines.append(f"【后区】实际平均 {mean_b_actual:.3f}  vs  理论随机 {mean_b_theo:.3f}")
        lines.append(f"【最佳】最高单期前区命中: {detail_df['最佳前区命中'].max()}")
    else:
        if grouped_pls:
            lines.append(f"【玩法】{effective_play}；仅按号码多重集统计整组命中。")
            lines.append(f"【整组】实际命中率 {group_rate_actual:.2%}  vs  理论随机 {group_rate_theo:.8%}")
        else:
            lines.append(f"【位置】实际平均命中 {mean_pos_actual:.3f}  vs  理论随机 {mean_pos_theo:.3f}")
            lines.append(f"【整组】实际命中率 {group_rate_actual:.2%}  vs  理论随机 {group_rate_theo:.8%}")
            lines.append(f"【最佳】最高单期位置命中: {detail_df['最佳位置命中'].max()}")
    lines.append("-" * 60)

    # 相对倾向分只用于排序观察，不是联合中奖概率或概率校准。
    if calib_points:
        probs = np.array([p[0] for p in calib_points])
        hits = np.array([p[1] for p in calib_points])
        order = np.argsort(probs)
        probs_s, hits_s = probs[order], hits[order]
        n = len(probs_s)
        bins = max(1, min(4, n // 5)) if n >= 5 else 1
        calib_rows = []
        for b in range(bins):
            lo = int(b * n / bins)
            hi = int((b + 1) * n / bins)
            seg_p = probs_s[lo:hi]
            seg_h = hits_s[lo:hi]
            if len(seg_p) == 0:
                continue
            calib_rows.append((float(seg_p.mean()), float(seg_h.mean()), len(seg_p)))
        if calib_rows:
            lines.append("【相对倾向分分组对照】（仅观察排序关系，不是概率校准）")
            for cp, ch, cnt in calib_rows:
                lines.append(f"  相对倾向分≈{cp:.4f} → 平均命中 {ch:.2f}  ({cnt} 组)")
            cp_arr = np.array([r[0] for r in calib_rows])
            ch_arr = np.array([r[1] for r in calib_rows])
            if len(cp_arr) >= 2 and np.std(cp_arr) > 0 and np.std(ch_arr) > 0:
                corr = float(np.corrcoef(cp_arr, ch_arr)[0, 1])
                verdict = "正相关" if corr > 0.1 else ("负相关" if corr < -0.1 else "弱相关")
                lines.append(f"  倾向分-命中相关系数: {corr:+.3f}  ({verdict})")

    lines.append("=" * 60)
    lines.append("⚠️ 样本量有限时统计波动大；彩票为随机事件，结果仅供参考。")

    # 校准图
    fig = None
    if calib_points and len(calib_points) >= 5:
        # 配置中文字体，避免缺字形告警
        import matplotlib.font_manager as fm
        font_candidates = ["Microsoft YaHei", "SimHei", "SimSun",
                           "Source Han Sans CN", "Noto Sans CJK SC"]
        available = {f.name for f in fm.fontManager.ttflist}
        for fname in font_candidates:
            if fname in available:
                plt.rcParams["font.sans-serif"] = [fname]
                plt.rcParams["axes.unicode_minus"] = False
                break
        fig, ax = plt.subplots(figsize=(8, 5))
        probs_all = np.array([p[0] for p in calib_points])
        hits_all = np.array([p[1] for p in calib_points])
        ax.scatter(probs_all, hits_all, alpha=0.3, s=20, color="#3498db")
        if calib_rows:
            cp_arr = np.array([r[0] for r in calib_rows])
            ch_arr = np.array([r[1] for r in calib_rows])
            ax.plot(cp_arr, ch_arr, "o-", color="#e74c3c", linewidth=2,
                    markersize=8, label="分桶均值")
        ax.set_xlabel("相对倾向分 (prob)")
        ax.set_ylabel("实际命中数")
        ax.set_title(f"{game_name} 相对倾向分与命中对照")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    return "\n".join(lines), detail_df, fig
