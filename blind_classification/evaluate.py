"""Evaluate blind classification results: accuracy, confusion, chance tests.

Reads the facet catalog from each result's manifest, so both 5-class and
30-class runs are handled. For 30-class runs an additional domain-level
accuracy is reported (was the chosen facet at least in the right Big Five
domain?).

Usage:
    python -X utf8 blind_classification/evaluate.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def binomial_tail(n: int, k: int, p: float) -> float:
    """P(X >= k) under Binomial(n, p)."""
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return total


def evaluate_path(path: Path) -> dict:
    label = path.stem[: -len("_classifications")]
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest = data["manifest"]
    facet_ids: list[str] = manifest["facet_ids"]
    results = data["results"]
    n = len(results)
    valid = [r for r in results if r["predicted_facet"] is not None]
    n_valid = len(valid)
    hits = [r for r in valid if r["predicted_facet"] == r["true_facet"]]
    accuracy = len(hits) / n_valid if n_valid else 0.0

    confusion = {f: {g: 0 for g in facet_ids} for f in facet_ids}
    for r in valid:
        confusion[r["true_facet"]][r["predicted_facet"]] += 1

    # Facets that actually appear in the data (as truth or prediction).
    used = sorted(
        {
            fid
            for r in valid
            for fid in (r["true_facet"], r["predicted_facet"])
            if fid in facet_ids
        }
    )

    per_facet = {}
    for f in facet_ids:
        true_count = sum(1 for r in valid if r["true_facet"] == f)
        pred_count = sum(1 for r in valid if r["predicted_facet"] == f)
        tp = confusion[f][f]
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / true_count if true_count else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_facet[f] = {
            "true_count": true_count,
            "predicted_count": pred_count,
            "correct": tp,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    chance = 1 / len(facet_ids)

    domain_hits = domain_valid = 0
    if any("_" in fid for fid in facet_ids):
        for r in valid:
            if "_" not in r["true_facet"] or "_" not in r["predicted_facet"]:
                continue
            domain_valid += 1
            if r["true_facet"].split("_")[0] == r["predicted_facet"].split("_")[0]:
                domain_hits += 1
    domain_accuracy = domain_hits / domain_valid if domain_valid else None

    return {
        "label": label,
        "manifest": manifest,
        "facet_ids": facet_ids,
        "n_items": n,
        "n_valid": n_valid,
        "n_failed": n - n_valid,
        "accuracy": round(accuracy, 4),
        "chance": chance,
        "binomial_p_vs_chance": binomial_tail(n_valid, len(hits), chance),
        "domain_accuracy": round(domain_accuracy, 4)
        if domain_accuracy is not None
        else None,
        "confusion": confusion,
        "per_facet": per_facet,
        "used_facets": used,
        "errors": [
            {
                "item_id": r["item_id"],
                "true_facet": r["true_facet"],
                "predicted_facet": r["predicted_facet"],
                "reason": r["reason"],
            }
            for r in valid
            if r["predicted_facet"] != r["true_facet"]
        ],
    }


def render_markdown(*evals: dict) -> str:
    lines = []
    lines.append("# 盲法构念分类评估报告\n")

    for ev in evals:
        lines.append(f"## {ev['label']}\n")
        lines.append(
            f"- 有效分类：{ev['n_valid']}/{ev['n_items']} "
            f"（失败 {ev['n_failed']}）"
        )
        lines.append(
            f"- 候选目录：{len(ev['facet_ids'])} 个 facet，"
            f"随机基线 {ev['chance']:.2%}"
        )
        lines.append(
            f"- 总体准确率：**{ev['accuracy']:.2%}**"
            f"（二项检验 p = {ev['binomial_p_vs_chance']:.4g}）"
        )
        if ev["domain_accuracy"] is not None:
            lines.append(f"- 域级准确率：**{ev['domain_accuracy']:.2%}**")
        lines.append("")

        lines.append("### 出现过的 facet 精确率 / 召回率\n")
        lines.append("| facet | 真值数 | 预测数 | 正确 | P | R | F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for f in ev["used_facets"]:
            p = ev["per_facet"][f]
            lines.append(
                f"| {f} | {p['true_count']} | {p['predicted_count']} "
                f"| {p['correct']} | {p['precision']} | {p['recall']} "
                f"| {p['f1']} |"
            )
        lines.append("")

        if len(ev["facet_ids"]) <= 8:
            lines.append("### 混淆矩阵（行=真值，列=预测）\n")
            lines.append("| 真值\\预测 | " + " | ".join(ev["facet_ids"]) + " |")
            lines.append("|---|" + "---|" * len(ev["facet_ids"]))
            for f in ev["facet_ids"]:
                row = ev["confusion"][f]
                lines.append(
                    f"| {f} | "
                    + " | ".join(str(row[g]) for g in ev["facet_ids"])
                    + " |"
                )
        else:
            lines.append("### 非零混淆对（真值 -> 预测：次数）\n")
            for f in ev["used_facets"]:
                row = ev["confusion"][f]
                for g, count in row.items():
                    if count and g != f:
                        lines.append(f"- {f} -> {g}：{count}")
            if not any(
                count
                for f in ev["used_facets"]
                for g, count in ev["confusion"][f].items()
                if count and g != f
            ):
                lines.append("- 无（所有预测与真值一致）")
        lines.append("")

        if ev["errors"]:
            lines.append("### 错分类明细\n")
            for err in ev["errors"]:
                lines.append(
                    f"- {err['item_id']}：真值 {err['true_facet']} "
                    f"→ 预测 {err['predicted_facet']}。"
                    f"理由：{err['reason']}"
                )
            lines.append("")

    lines.append("## 使用边界\n")
    lines.append(
        "- 分类器是 LLM 而非人类专家；结果只能作为开发期语义证据，"
        "不能替代真人盲法评判。"
    )
    lines.append(
        "- 金标准准确率用于校准分类器能力；只有金标准表现可信时，"
        "对本次题目的分类结果才有参考价值。"
    )
    lines.append(
        "- 高分类准确率证明题目文本语义与目标构念对应（内容效度），"
        "不证明被试作答会按该构念运作。"
    )
    return "\n".join(lines)


def main() -> None:
    paths = sorted(RESULTS_DIR.glob("*_classifications.json"))

    def group_key(path: Path) -> int:
        stem = path.stem[: -len("_classifications")]
        if stem.startswith("gold"):
            return 0
        if stem.startswith("our"):
            return 1
        return 2

    paths.sort(key=lambda p: (group_key(p), p.stem))
    evals = [evaluate_path(p) for p in paths]
    report = render_markdown(*evals)
    out = RESULTS_DIR / "evaluation_report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n报告已写入 {out}")


if __name__ == "__main__":
    main()
