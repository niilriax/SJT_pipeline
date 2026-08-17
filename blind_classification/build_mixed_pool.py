"""Build a source-separated mixed analysis from existing blind results.

Each classification call is independent and receives only the facet catalog,
the item's situation and options.  Therefore the already completed runs can
be pooled without exposing source labels to the classifier.  This script does
not invent or reclassify any item; it only adds local source metadata and
recomputes the combined evaluation.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blind_classification.mixed_classify import evaluate_mixed


RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "mixed_mussel_compliance_gregariousness_c30"


SOURCES = [
    ("mussel", "Mussel 金标准", "gold_mussel_c30_classifications.json"),
    (
        "compliance",
        "compliance 正式题库",
        "our_compliance_items_c30_classifications.json",
    ),
    (
        "gregariousness",
        "gregariousness 冻结题库",
        "bank_bank-6a0f832c-cc2-v1-ebf4fddc463e_c30_classifications.json",
    ),
]


def main() -> None:
    rows: list[dict] = []
    source_counts: Counter[str] = Counter()
    manifests: list[dict] = []
    for source_group, source_label, filename in SOURCES:
        path = RESULTS_DIR / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        manifests.append(data["manifest"])
        for result in data["results"]:
            enriched = dict(result)
            enriched["source_group"] = source_group
            enriched["source_label"] = source_label
            rows.append(enriched)
            source_counts[source_group] += 1

    random.Random(20260815).shuffle(rows)
    evaluation = evaluate_mixed(rows)
    evaluation["analysis_mode"] = "pooled_existing_independent_blind_calls"
    evaluation["source_labels_sent_to_classifier"] = False
    evaluation["source_counts"] = dict(source_counts)
    evaluation["source_files"] = [filename for _, _, filename in SOURCES]
    evaluation["note"] = (
        "每题独立调用，提示词只含共享 facet 目录、情境和选项；"
        "来源标签仅在本地合并后用于分组统计。"
    )

    manifest = {
        "task": "blind_mixed_source_facet_classification",
        "label": OUTPUT_STEM,
        "catalog_size": 30,
        "item_count": len(rows),
        "source_counts": dict(source_counts),
        "source_labels_sent_to_classifier": False,
        "analysis_mode": "pooled_existing_independent_blind_calls",
        "mixed_shuffle_seed": 20260815,
        "component_manifests": [
            {
                "label": m.get("label"),
                "model_id": m.get("model_id"),
                "prompt_sha256": m.get("prompt_sha256"),
                "generated_at": m.get("generated_at"),
            }
            for m in manifests
        ],
    }

    result_path = RESULTS_DIR / f"{OUTPUT_STEM}_classifications.json"
    result_path.write_text(
        json.dumps(
            {"manifest": manifest, "results": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    eval_path = RESULTS_DIR / f"{OUTPUT_STEM}_evaluation.json"
    eval_path.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"混合结果已写入: {result_path}")
    print(f"混合评估已写入: {eval_path}")
    for source in evaluation["source_order"]:
        row = evaluation["by_source"][source]
        print(
            f"{source}: {row['n_valid']}/{row['n_total']}, "
            f"top1={row['top1_accuracy']:.1%}, "
            f"domain={row['domain_accuracy']:.1%}"
        )


if __name__ == "__main__":
    main()
