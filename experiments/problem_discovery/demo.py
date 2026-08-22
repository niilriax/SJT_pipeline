"""在真实运行数据上演示序贯贝叶斯问题发现。

数据源（真实产物，非合成）：
- outputs/virtual_responses/<run>/<bank>/psychometrics/
  item_quality.csv、option_statistics.csv、scored_sjt_responses.csv、
  scored_neo_ffi_responses.csv
- blind_classification/results/our_compliance_items_c30_classifications.json

流程：对每道题提取 findings（确定性阈值）→ 贝叶斯后验 →
三带决策（healthy / act / ambiguous）→ 歧义时按信息增益选下一指标，
若选中的是 OPTION_PBS 则当场从真实作答数据计算并重新决策。
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.problem_discovery.diagnoser import decide  # noqa: E402
from experiments.problem_discovery.taxonomy import (  # noqa: E402
    FINDINGS,
    PROBLEMS,
)

RUN_ID = "f4cefb8f-71cf-4bbb-aa72-c426446979fc"
BANK = "bank-v2-75babf8179a9"
PSYCH = ROOT / "outputs" / "virtual_responses" / RUN_ID / BANK / "psychometrics"
BLIND_PATH = (
    ROOT / "blind_classification" / "results"
    / "our_compliance_items_c30_classifications.json"
)
OUT_DIR = Path(__file__).resolve().parent / "out"

# PBS 判定阈值：选择者 ≥5 人且 r < 该值 → F_OPTION_NEG_PBS。
PBS_FLAG_R = -0.15
PBS_MIN_N = 5
ORDER_TOLERANCE = 0.05


def load_quality() -> dict[str, dict]:
    rows = {}
    with open(PSYCH / "item_quality.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rows[row["item_id"]] = {
                "citc_r": float(row["citc_r"]),
                "difficulty": float(row["difficulty"]),
                "effective_option_count": int(row["effective_option_count"]),
                "minimum_option_rate": float(row["minimum_option_rate"]),
            }
    return rows


def load_blind_map(quality_ids: set[str]) -> dict[str, tuple[str, str | None]]:
    """our_Q00X 按题号映射到 row-0X；返回 {item_id: (预测facet, finding_code|None)}。"""
    data = json.loads(BLIND_PATH.read_text(encoding="utf-8"))
    by_number: dict[int, dict] = {}
    for result in data["results"]:
        number = int(result["item_id"].removeprefix("our_Q"))
        by_number[number] = result
    mapping: dict[str, tuple[str, str | None]] = {}
    for item_id in sorted(quality_ids):
        match = re.search(r"row-(\d+)", item_id)
        if not match:
            continue
        row_number = int(match.group(1))
        result = by_number.get(row_number)
        if result is None:
            continue
        predicted = result["predicted_facet"]
        true = result["true_facet"]
        code = None
        if predicted != true:
            same_domain = true.split("_")[0] == predicted.split("_")[0]
            code = "F_BLIND_SAME" if same_domain else "F_BLIND_CROSS"
        mapping[item_id] = (predicted, code)
    return mapping


def load_neo_a_means() -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    with open(PSYCH / "scored_neo_ffi_responses.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["dimension_code"] == "A":
                values[row["respondent_id"]].append(float(row["score"]))
    return {rid: sum(vals) / len(vals) for rid, vals in values.items()}


def load_option_scores() -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    with open(PSYCH / "option_statistics.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            scores[row["item_id"]][row["option_id"]] = float(row["score"])
    return scores


def load_selection_vectors(
    quality_ids: set[str],
    neo_means: dict[str, float],
) -> dict[str, dict[str, dict[str, float]]]:
    """{item_id: {option_id: {respondent_id: 选择者 Neo-A 均值}}}。"""
    selectors: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with open(PSYCH / "scored_sjt_responses.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            item_id = row["item_id"]
            if item_id not in quality_ids or row["respondent_id"] not in neo_means:
                continue
            selectors[item_id][row["selected_option_id"]][row["respondent_id"]] = (
                neo_means[row["respondent_id"]]
            )
    return dict(selectors)


def compute_option_order(
    item_id: str,
    option_scores: dict[str, dict[str, float]],
    selectors: dict[str, dict[str, dict[str, float]]],
) -> list[str]:
    """选项得分更高者，其选择者的 Neo-A 均值反而更低 → 次序颠倒。"""
    flagged: list[str] = []
    options = list(option_scores.get(item_id, {}))
    for left in options:
        for right in options:
            if left == right:
                continue
            left_vals = list(selectors.get(item_id, {}).get(left, {}).values())
            right_vals = list(selectors.get(item_id, {}).get(right, {}).values())
            if len(left_vals) < PBS_MIN_N or len(right_vals) < PBS_MIN_N:
                continue
            score_left = option_scores[item_id][left]
            score_right = option_scores[item_id][right]
            mean_left = sum(left_vals) / len(left_vals)
            mean_right = sum(right_vals) / len(right_vals)
            if score_left > score_right and mean_left + ORDER_TOLERANCE < mean_right:
                flagged.append(
                    f"选项{left}(分{score_left:.0f},A均{mean_left:.3f})"
                    f" 低于 选项{right}(分{score_right:.0f},A均{mean_right:.3f})"
                )
    return flagged


def load_rest_by_respondent() -> dict[tuple[str, str], float]:
    """返回 {(item_id, respondent_id): 该被试在该题以外 15 题的总分}。"""
    per_respondent: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with open(PSYCH / "scored_sjt_responses.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            per_respondent[row["respondent_id"]].append(
                (row["item_id"], float(row["score"]))
            )
    rest: dict[tuple[str, str], float] = {}
    for respondent, pairs in per_respondent.items():
        total = sum(score for _, score in pairs)
        for item_id, score in pairs:
            rest[(item_id, respondent)] = total - score
    return rest


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def compute_option_pbs(
    item_id: str,
    option_scores: dict[str, dict[str, float]],
    selectors: dict[str, dict[str, dict[str, float]]],
    rest_by_respondent: dict[tuple[str, str], float],
) -> list[tuple[str, str]]:
    """符号感知的点二列相关（选择(1/0) × 分面其余总分）。

    健康方向：低于题均分的选项应负相关、高于题均分应正相关。
    只有方向与得分相反才是问题：
    - 高分选项 r_pb < -0.15 → F_OPTION_HIGH_NEG_PBS（高分不吸引高构念者）
    - 低分选项 r_pb > +0.15 → F_OPTION_LOW_POS_PBS（低分选项吸引高构念者）
    返回 [(finding_code, detail), ...]。
    """
    flagged: list[tuple[str, str]] = []
    respondent_ids = sorted(
        {rid for select_map in selectors.get(item_id, {}).values() for rid in select_map}
    )
    if len(respondent_ids) < 10:
        return flagged
    rest = [rest_by_respondent.get((item_id, rid), 0.0) for rid in respondent_ids]
    # 以选择人数加权的题均分，作为“高低分”的分界线
    weighted_sum = 0.0
    weighted_n = 0
    for option_id, select_map in selectors.get(item_id, {}).items():
        score = float(option_scores.get(item_id, {}).get(option_id, 0.0))
        weighted_sum += score * len(select_map)
        weighted_n += len(select_map)
    if weighted_n == 0:
        return flagged
    mean_score = weighted_sum / weighted_n

    for option_id, select_map in selectors.get(item_id, {}).items():
        if len(select_map) < PBS_MIN_N:
            continue
        score = float(option_scores.get(item_id, {}).get(option_id, 0.0))
        binary = [1.0 if rid in select_map else 0.0 for rid in respondent_ids]
        r = _pearson(binary, rest)
        if r is None:
            continue
        detail = f"选项{option_id}(分{score:.0f},r_pb={r:.3f})"
        if score > mean_score and r < PBS_FLAG_R:
            flagged.append(("F_OPTION_HIGH_NEG_PBS", detail))
        elif score < mean_score and r > -PBS_FLAG_R:
            flagged.append(("F_OPTION_LOW_POS_PBS", detail))
    return flagged


def extract_findings(
    item_id: str,
    item: dict,
    blind_map: dict,
    option_order: dict,
) -> tuple[dict[str, str], set[str]]:
    fired: dict[str, str] = {}
    r = item["citc_r"]
    difficulty = item["difficulty"]
    if r < 0:
        fired["F_CITC_NEG"] = f"citc={r:.3f}"
    elif r < 0.20:
        fired["F_CITC_LOW"] = f"citc={r:.3f}"
    if difficulty > 0.80:
        fired["F_DIFF_HIGH"] = f"difficulty={difficulty:.3f}"
    elif difficulty < 0.20:
        fired["F_DIFF_LOW"] = f"difficulty={difficulty:.3f}"
    if item["effective_option_count"] < 3:
        fired["F_OPTION_FEW"] = f"有效选项={item['effective_option_count']}"
    if item["minimum_option_rate"] == 0:
        fired["F_OPTION_ZERO"] = "min_rate=0.00"
    blind = blind_map.get(item_id)
    if blind and blind[1]:
        fired[blind[1]] = f"盲法预测={blind[0]}"
    for detail in option_order.get(item_id, []):
        fired["F_OPTION_ORDER_BROKEN"] = detail
    observed = {"CITC", "DIFF", "OPTION", "CRITERION_ORDER"}
    if item_id in blind_map:
        observed.add("BLIND")
    return fired, observed


def render_row(
    item_id: str,
    result: dict,
    fired: dict[str, str],
    rounds: int,
) -> list[str]:
    top_problem, top_value = result["ranked"][0]
    runner_problem, runner_value = result["ranked"][1]
    codes = ",".join(sorted(fired))
    return [
        item_id,
        codes,
        f"{top_problem} ({top_value:.2f})",
        f"{runner_problem} ({runner_value:.2f})",
        result["decision"],
        result["action"],
        result["suggestion"] or "",
        str(rounds),
    ]


def main() -> None:
    quality = load_quality()
    blind_map = load_blind_map(set(quality))
    print(f"[demo] 题库: {BANK}；题目数 {len(quality)}；"
          f"盲法映射 {len(blind_map)} 题")
    print(f"[demo] 假设: blind 的 our_Q00X 对应 row-0X "
          f"（2026-08-15 同批次 compliance 16 题）")

    neo_means = load_neo_a_means()
    option_scores = load_option_scores()
    selectors = load_selection_vectors(set(quality), neo_means)
    rest_by_respondent = load_rest_by_respondent()
    option_order: dict[str, list[str]] = {}
    for item_id in quality:
        option_order[item_id] = compute_option_order(
            item_id, option_scores, selectors
        )

    header = ["item_id", "findings", "top", "runner", "decision",
              "action", "suggestion", "rounds"]
    rows = []
    final_fired: dict[str, dict[str, str]] = {}
    for item_id in sorted(quality):
        item = quality[item_id]
        fired, observed = extract_findings(
            item_id, item, blind_map, option_order
        )
        result = decide(fired, observed)
        rounds = 1
        while result["action"].startswith("measure:") and rounds < 3:
            family = result["action"].split(":", 1)[1]
            if family == "OPTION_PBS":
                for code, detail in compute_option_pbs(
                    item_id, option_scores, selectors, rest_by_respondent
                ):
                    fired[code] = detail
                observed.add("OPTION_PBS")
            elif family == "ALPHA_DROP":
                # 该指标当前流水线未导出；保持未观察并停止追问。
                result["action"] = "INVESTIGATE"
                result["suggestion"] = "alpha_if_deleted 未导出，需补指标"
                break
            result = decide(fired, observed)
            rounds += 1
        final_fired[item_id] = dict(fired)
        rows.append(render_row(item_id, result, fired, rounds))

    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]
    def fmt(row: list[str]) -> str:
        return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))

    print("\n" + fmt(header))
    print("-" * (sum(widths) + 2 * (len(header) - 1)))
    for row in rows:
        print(fmt(row))

    ambiguous = [r for r in rows if r[4] == "ambiguous"]
    acts = [r for r in rows if r[4] == "act"]
    healthy = [r for r in rows if r[4] == "healthy"]
    print(f"\n[demo] healthy={len(healthy)} 明确问题={len(acts)} "
          f"歧义升级={len(ambiguous)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"problem_register_{BANK}.csv"
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[demo] 问题登记表已写入 {out_path}")

    print("\n[demo] 各题触发 finding 明细（可审计：每个数值 → 阈值 → 代码）")
    for item_id in sorted(quality):
        fired = final_fired[item_id]
        if not fired:
            continue
        print(f"  {item_id}")
        for code, detail in fired.items():
            print(f"    {code} [{FINDINGS[code][1]}]: {detail}")


if __name__ == "__main__":
    main()
