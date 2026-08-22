"""Blind facet classification of SJT items.

Gold set (classifier calibration):
    Mussel et al. (2018) items, 5 facets x 22 items, from docs/mussel_zh.json.
    Facet labels are known ground truth and are NEVER shown to the classifier.

Test sets:
    - The compliance production run's final test items (16 items) from the
      respondent form.
    - Any frozen item bank via --bank RUN_ID BANK_ID.

Blindness: each prompt contains only the candidate facet catalog (equal,
unlabeled candidates) plus one item's situation and options. Item ids, true
facet labels, scoring keys, behavioral levels and rationales never leave the
local machine.

Usage:
    python -X utf8 blind_classification/classify.py --catalog-size 30
    python -X utf8 blind_classification/classify.py --catalog-size 5
    python -X utf8 blind_classification/classify.py --skip-gold --skip-ours \
        --catalog-size 30 --bank RUN_ID BANK_ID
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from blind_classification.catalog import (
    ALL_FACET_IDS,
    FACET_IDS_5,
    MUSSEL_FACET_KEYS,
    MUSSEL_KEY_TO_REGISTRY_FACET,
    REGISTRY_TO_SHORT_FACET,
    build_catalog,
    catalog_text,
)
from mussel_evidence_extraction import load_mussel_items
from sjt_system.agent.client import get_model, with_compatible_structured_output

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_ROOT = (
    Path(__file__).resolve().parents[1] / "outputs" / "run_checkpoints"
)
FINAL_TEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "final_reports"
    / "6b94c430-8ed2-4ad4-bbc7-1916ff6731d0"
    / "bank-v3-7a37ddbdfe19"
    / "final_test.json"
)
MAX_ATTEMPTS = 3


def facet_ids_for(catalog_size: int) -> list[str]:
    if catalog_size == 30:
        return list(ALL_FACET_IDS)
    if catalog_size == 5:
        return list(FACET_IDS_5)
    raise ValueError("catalog_size 只能是 5 或 30")


class ClassificationResult(BaseModel):
    """One blind classification decision."""

    facet_id: str = Field(description="Single best-matching facet id")
    reason: str = Field(
        description="Brief explanation of the behavioral contrast and "
        "why the chosen facet predicts it best"
    )


def build_classification_prompt(
    catalog: list[dict[str, Any]],
    item: dict[str, Any],
    facet_ids: list[str],
) -> str:
    """One item per call; catalog + instructions + blind item text."""

    option_lines = []
    options = item["options"]
    if isinstance(options, dict):
        for letter, text in options.items():
            option_lines.append(f"{letter}. {text}")
    else:
        for option in options:
            option_lines.append(f"{option['option_id']}. {option['text']}")

    allowed = "、".join(facet_ids)
    return (
        "你是一名人格测验评审。下面给出 "
        f"{len(facet_ids)} 个人格特质（facet）的定义，"
        "以及一道情境判断题（情境 + 四个行为选项）。\n\n"
        "请判断：这道题的行为对照（高特质与低特质的人会做出不同选择）"
        "最可能由哪一个人格特质驱动？\n\n"
        "# 人格特质目录\n\n"
        f"{catalog_text(catalog)}\n\n"
        "# 判断规则\n\n"
        "1. 只根据行为对照判断，不要被表面词句误导。\n"
        "2. 关键问题：哪个特质的高分者与低分者会在这道题上做出最一致的"
        "不同选择？\n"
        "3. 先比较行为对照更符合哪个维度（宜人性/尽责性/外向性/神经质/"
        "开放性），再在维度内部选择解释力最强、最排他的那一个 facet。\n"
        "4. 必须从上面的目录中单选一个 facet，不允许回答“不确定”。\n\n"
        "# 输出格式\n\n"
        "只返回一个 JSON 对象："
        '{"facet_id":"在 ' + allowed + ' 中选一个","reason":"简短说明判断依据"}。\n\n'
        "# 题目\n\n"
        f"情境：{item['situation']}\n\n"
        "选项：\n" + "\n".join(option_lines)
    )


def load_gold_items(catalog_size: int) -> list[dict[str, Any]]:
    """Mussel gold items with known facet labels."""

    items: list[dict[str, Any]] = []
    for key in MUSSEL_FACET_KEYS:
        registry_id = MUSSEL_KEY_TO_REGISTRY_FACET[key]
        label = (
            registry_id if catalog_size == 30 else REGISTRY_TO_SHORT_FACET[registry_id]
        )
        for raw in load_mussel_items(key):
            items.append(
                {
                    "item_id": raw["item_id"],
                    "facet": label,
                    "situation": raw["situation"],
                    "options": raw["options"],
                }
            )
    return items


def load_our_items(catalog_size: int) -> list[dict[str, Any]]:
    """The production run's final test items, stripped to blind text only."""

    data = json.loads(FINAL_TEST_PATH.read_text(encoding="utf-8"))
    form_items = data["respondent_form"]["items"]
    label = "agreeableness_compliance" if catalog_size == 30 else "compliance"
    items: list[dict[str, Any]] = []
    for entry in form_items:
        items.append(
            {
                "item_id": f"our_{entry['public_item_id']}",
                "facet": label,
                "situation": entry["scenario"],
                "options": [
                    {
                        "option_id": option["option_id"],
                        "text": option["text"],
                    }
                    for option in entry["response_options"]
                ],
            }
        )
    return items


def load_bank_items(
    run_id: str,
    bank_id: str,
    catalog_size: int,
) -> list[dict[str, Any]]:
    """A frozen item bank from a run checkpoint, stripped to blind text only."""

    checkpoint_path = CHECKPOINT_ROOT / f"{run_id}.json"
    data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    state = data["state"]
    frozen = state.get("frozen_item_bank") or []
    items: list[dict[str, Any]] = []
    for entry in frozen:
        dimension = str(entry.get("target_dimension_id") or "")
        if catalog_size == 30:
            label = dimension if dimension in ALL_FACET_IDS else dimension
        else:
            label = REGISTRY_TO_SHORT_FACET.get(dimension, dimension)
        items.append(
            {
                "item_id": f"bank_{entry['item_id']}",
                "facet": label,
                "situation": entry["scenario"],
                "options": [
                    {
                        "option_id": option["option_id"],
                        "text": option["text"],
                    }
                    for option in entry["response_options"]
                ],
            }
        )
    return items


async def classify_one(
    runnable: Any,
    catalog: list[dict[str, Any]],
    item: dict[str, Any],
    *,
    facet_ids: list[str],
    semaphore: asyncio.Semaphore,
    item_index: int,
    total: int,
) -> dict[str, Any]:
    async with semaphore:
        prompt = build_classification_prompt(catalog, item, facet_ids)
        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                result = await runnable.ainvoke(prompt)
                predicted = str(result.get("facet_id") or "")
                if predicted not in facet_ids:
                    raise ValueError(
                        f"预测 facet_id {predicted!r} 不在候选目录中"
                    )
                print(
                    f"  [{item_index}/{total}] {item['item_id']} "
                    f"-> {predicted}",
                    flush=True,
                )
                return {
                    "item_id": item["item_id"],
                    "true_facet": item["facet"],
                    "predicted_facet": predicted,
                    "reason": result["reason"],
                    "attempt": attempt,
                }
            except Exception as exc:  # noqa: BLE001 - retry any transient failure
                last_error = exc
                if attempt >= MAX_ATTEMPTS:
                    break
                print(
                    f"  [{item_index}/{total}] {item['item_id']} "
                    f"第 {attempt} 次失败：{exc}；重试中",
                    flush=True,
                )
        print(
            f"  [{item_index}/{total}] {item['item_id']} 分类失败：{last_error}",
            flush=True,
        )
        return {
            "item_id": item["item_id"],
            "true_facet": item["facet"],
            "predicted_facet": None,
            "reason": f"classification failed after {MAX_ATTEMPTS} attempts: {last_error}",
            "attempt": MAX_ATTEMPTS,
        }


async def run_classification(
    model_id: str,
    concurrency: int,
    items: list[dict[str, Any]],
    *,
    catalog_size: int,
    label: str,
    seed: int = 20260814,
) -> list[dict[str, Any]]:
    facet_ids = facet_ids_for(catalog_size)
    catalog = build_catalog(facet_ids)
    model = get_model(
        model_id,
        thinking_type="enabled",
        reasoning_effort="high",
    )
    runnable, method = with_compatible_structured_output(
        model, ClassificationResult
    )
    ordered = list(items)
    random.Random(seed).shuffle(ordered)
    semaphore = asyncio.Semaphore(concurrency)
    print(f"[{label}] {len(ordered)} 题，模型 {model_id}，输出方式 {method}")
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
    prompt = build_classification_prompt(catalog, ordered[0], facet_ids)
    manifest = {
        "task": "blind_facet_classification",
        "label": label,
        "model_id": model_id,
        "structured_output_method": method,
        "temperature": None,
        "thinking": "enabled",
        "reasoning_effort": "high",
        "catalog_size": catalog_size,
        "facet_ids": facet_ids,
        "item_count": len(ordered),
        "shuffle_seed": seed,
        "order": [item["item_id"] for item in ordered],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{label}_classifications.json"
    out_path.write_text(
        json.dumps(
            {"manifest": manifest, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[{label}] 结果已写入 {out_path}")
    return results


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-pro-guan")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--catalog-size", type=int, default=30, choices=[5, 30])
    parser.add_argument("--skip-gold", action="store_true")
    parser.add_argument("--skip-ours", action="store_true")
    parser.add_argument(
        "--bank",
        nargs=2,
        metavar=("RUN_ID", "BANK_ID"),
        help="对指定运行的冻结题库分类，参数为 run_id 和 item_bank_id",
    )
    args = parser.parse_args()
    suffix = f"c{args.catalog_size}"

    if not args.skip_gold:
        await run_classification(
            args.model,
            args.concurrency,
            load_gold_items(args.catalog_size),
            catalog_size=args.catalog_size,
            label=f"gold_mussel_{suffix}",
        )
    if not args.skip_ours:
        await run_classification(
            args.model,
            args.concurrency,
            load_our_items(args.catalog_size),
            catalog_size=args.catalog_size,
            label=f"our_compliance_items_{suffix}",
        )
    if args.bank:
        run_id, bank_id = args.bank
        items = load_bank_items(run_id, bank_id, args.catalog_size)
        await run_classification(
            args.model,
            args.concurrency,
            items,
            catalog_size=args.catalog_size,
            label=f"bank_{bank_id}_{suffix}",
        )


if __name__ == "__main__":
    asyncio.run(main())
