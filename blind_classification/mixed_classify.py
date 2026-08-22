"""Blindly classify a mixed bank of Mussel and generated SJT items.

The classifier receives only the shared 30-facet catalog plus each item's
situation and options.  Source labels are retained locally for evaluation
and are never included in the prompt.

Usage:
    python -X utf8 blind_classification/mixed_classify.py
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blind_classification.classify import (
    RESULTS_DIR,
    build_classification_prompt,
    classify_one,
    facet_ids_for,
    load_bank_items,
    load_gold_items,
    load_our_items,
)
from blind_classification.catalog import build_catalog
from sjt_system.agent.client import get_model, with_compatible_structured_output
from blind_classification.classify import ClassificationResult


GREGARIOUSNESS_RUN_ID = "6a0f832c-cc2b-46ca-a074-e449abf9ba67"
GREGARIOUSNESS_BANK_ID = "bank-6a0f832c-cc2-v1-ebf4fddc463e"
OUTPUT_STEM = "mixed_mussel_compliance_gregariousness_c30"


def build_mixed_items() -> list[dict]:
    """Load all items and attach local-only source metadata."""

    groups = [
        ("mussel", "Mussel 金标准", load_gold_items(30)),
        ("compliance", "compliance 正式题库", load_our_items(30)),
        (
            "gregariousness",
            "gregariousness 冻结题库",
            load_bank_items(
                GREGARIOUSNESS_RUN_ID,
                GREGARIOUSNESS_BANK_ID,
                30,
            ),
        ),
    ]
    items: list[dict] = []
    for source_group, source_label, group_items in groups:
        for item in group_items:
            enriched = dict(item)
            # These fields are deliberately ignored by build_classification_prompt.
            enriched["source_group"] = source_group
            enriched["source_label"] = source_label
            items.append(enriched)
    return items


def evaluate_mixed(results: list[dict]) -> dict:
    """Compute source-separated metrics from the blind classifications."""

    source_results: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        source_results[result["source_group"]].append(result)

    by_source: dict[str, dict] = {}
    for source, rows in source_results.items():
        valid = [r for r in rows if r["predicted_facet"] is not None]
        hits = [r for r in valid if r["predicted_facet"] == r["true_facet"]]
        domain_valid = [
            r
            for r in valid
            if "_" in str(r["true_facet"])
            and "_" in str(r["predicted_facet"])
        ]
        domain_hits = [
            r
            for r in domain_valid
            if r["true_facet"].split("_")[0]
            == r["predicted_facet"].split("_")[0]
        ]
        same_domain_facet_errors = [
            r
            for r in valid
            if r["predicted_facet"] != r["true_facet"]
            and "_" in str(r["true_facet"])
            and "_" in str(r["predicted_facet"])
            and r["true_facet"].split("_")[0]
            == r["predicted_facet"].split("_")[0]
        ]
        cross_domain_errors = [
            r
            for r in valid
            if r["predicted_facet"] != r["true_facet"]
            and r not in same_domain_facet_errors
        ]
        by_source[source] = {
            "source_label": rows[0]["source_label"],
            "n_total": len(rows),
            "n_valid": len(valid),
            "top1_correct": len(hits),
            "top1_accuracy": len(hits) / len(valid) if valid else None,
            "domain_valid": len(domain_valid),
            "domain_correct": len(domain_hits),
            "domain_accuracy": len(domain_hits) / len(domain_valid)
            if domain_valid
            else None,
            "same_domain_facet_errors": len(same_domain_facet_errors),
            "cross_domain_errors": len(cross_domain_errors),
            "errors": [
                {
                    "item_id": r["item_id"],
                    "true_facet": r["true_facet"],
                    "predicted_facet": r["predicted_facet"],
                    "same_domain": r in same_domain_facet_errors,
                    "reason": r["reason"],
                }
                for r in valid
                if r["predicted_facet"] != r["true_facet"]
            ],
        }
    valid_all = [r for r in results if r["predicted_facet"] is not None]
    top1_hits_all = [
        r for r in valid_all if r["predicted_facet"] == r["true_facet"]
    ]
    domain_valid_all = [
        r
        for r in valid_all
        if "_" in str(r["true_facet"])
        and "_" in str(r["predicted_facet"])
    ]
    domain_hits_all = [
        r
        for r in domain_valid_all
        if r["true_facet"].split("_")[0]
        == r["predicted_facet"].split("_")[0]
    ]
    return {
        "source_order": ["mussel", "compliance", "gregariousness"],
        "by_source": by_source,
        "n_total": len(results),
        "n_valid": sum(v["n_valid"] for v in by_source.values()),
        "overall": {
            "n_total": len(results),
            "n_valid": len(valid_all),
            "top1_correct": len(top1_hits_all),
            "top1_accuracy": len(top1_hits_all) / len(valid_all)
            if valid_all
            else None,
            "domain_valid": len(domain_valid_all),
            "domain_correct": len(domain_hits_all),
            "domain_accuracy": len(domain_hits_all) / len(domain_valid_all)
            if domain_valid_all
            else None,
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-pro-guan")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    items = build_mixed_items()
    facet_ids = facet_ids_for(30)
    catalog = build_catalog(facet_ids)
    model = get_model(
        args.model,
        thinking_type="enabled",
        reasoning_effort="high",
    )
    runnable, method = with_compatible_structured_output(model, ClassificationResult)
    ordered = list(items)
    random.Random(args.seed).shuffle(ordered)
    semaphore = asyncio.Semaphore(args.concurrency)
    print(
        f"[mixed] {len(ordered)} 题：Mussel 110 + compliance 16 + "
        f"gregariousness 16；模型 {args.model}；输出方式 {method}",
        flush=True,
    )
    results = await asyncio.gather(
        *[
            classify_one(
                runnable,
                catalog,
                item,
                facet_ids=facet_ids,
                semaphore=semaphore,
                item_index=index,
                total=len(ordered),
            )
            for index, item in enumerate(ordered, 1)
        ]
    )
    source_map = {
        item["item_id"]: (item["source_group"], item["source_label"])
        for item in ordered
    }
    for result in results:
        source_group, source_label = source_map[result["item_id"]]
        result["source_group"] = source_group
        result["source_label"] = source_label

    prompt = build_classification_prompt(catalog, ordered[0], facet_ids)
    manifest = {
        "task": "blind_mixed_source_facet_classification",
        "label": OUTPUT_STEM,
        "model_id": args.model,
        "structured_output_method": method,
        "temperature": None,
        "thinking": "enabled",
        "reasoning_effort": "high",
        "catalog_size": 30,
        "facet_ids": facet_ids,
        "item_count": len(ordered),
        "source_counts": dict(Counter(item["source_group"] for item in ordered)),
        "source_labels_sent_to_classifier": False,
        "shuffle_seed": args.seed,
        "order": [item["item_id"] for item in ordered],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    payload = {"manifest": manifest, "results": results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{OUTPUT_STEM}_classifications.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    evaluation = evaluate_mixed(results)
    evaluation_path = RESULTS_DIR / f"{OUTPUT_STEM}_evaluation.json"
    evaluation_path.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[mixed] 分类结果已写入 {result_path}")
    print(f"[mixed] 分组评估已写入 {evaluation_path}")
    for source in evaluation["source_order"]:
        row = evaluation["by_source"][source]
        print(
            f"  {source}: top1={row['top1_accuracy']:.1%}, "
            f"domain={row['domain_accuracy']:.1%}, "
            f"same_domain_facet_errors={row['same_domain_facet_errors']}, "
            f"cross_domain_errors={row['cross_domain_errors']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
