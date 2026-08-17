"""Virtual respondent pool loading and deterministic sampling."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


DEFAULT_POOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "virtual_respondents.json"
)
DEFAULT_SELECTION_SEED = 7
DEFAULT_MAX_CONCURRENCY = 5
MAX_ALLOWED_CONCURRENCY = 20
DEFAULT_MAX_RETRIES = 2
MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE = 30
PERSONA_MODE_SUMMARY_PLUS_ITEMS = "summary_plus_items"
SUPPORTED_PERSONA_MODES = (PERSONA_MODE_SUMMARY_PLUS_ITEMS,)
DEFAULT_PERSONA_MODES = SUPPORTED_PERSONA_MODES


@lru_cache(maxsize=4)
def _load_pool_cached(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    with path.open("r", encoding="utf-8") as handle:
        pool = json.load(handle)
    _validate_pool(pool)
    return pool


def _validate_pool(pool: object) -> None:
    if not isinstance(pool, dict):
        raise ValueError("虚拟被试池必须是对象")
    if pool.get("schema_version") != 1:
        raise ValueError("虚拟被试池 schema_version 必须为 1")

    items = pool.get("items")
    respondents = pool.get("respondents")
    response_scale = pool.get("response_scale")
    if not isinstance(items, list) or not items:
        raise ValueError("虚拟被试池缺少人格题目")
    if not isinstance(respondents, list) or not respondents:
        raise ValueError("虚拟被试池缺少被试记录")
    if not isinstance(response_scale, dict):
        raise ValueError("虚拟被试池缺少作答标签")
    if pool.get("item_count") != len(items):
        raise ValueError("虚拟被试池 item_count 与实际题目数不一致")
    if pool.get("respondent_count") != len(respondents):
        raise ValueError("虚拟被试池 respondent_count 与实际人数不一致")

    item_ids = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("虚拟被试池包含无效人格题目")
        item_id = item.get("item_id")
        statement = item.get("statement")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("人格题目缺少 item_id")
        if not isinstance(statement, str) or not statement:
            raise ValueError(f"人格题目 {item_id} 缺少题干")
        item_ids.append(item_id)
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("虚拟被试池包含重复人格题目 ID")

    respondent_ids = []
    for respondent in respondents:
        if not isinstance(respondent, dict):
            raise ValueError("虚拟被试池包含无效被试记录")
        respondent_id = respondent.get("respondent_id")
        values = respondent.get("response_values")
        if not isinstance(respondent_id, str) or not respondent_id:
            raise ValueError("虚拟被试记录缺少 respondent_id")
        if not isinstance(values, list) or len(values) != len(items):
            raise ValueError(
                f"虚拟被试 {respondent_id} 的人格作答数量不正确"
            )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            or value > 5
            for value in values
        ):
            raise ValueError(
                f"虚拟被试 {respondent_id} 包含不在 1-5 范围内的作答"
            )
        respondent_ids.append(respondent_id)
    if len(set(respondent_ids)) != len(respondent_ids):
        raise ValueError("虚拟被试池包含重复 respondent_id")

    for value in range(1, 6):
        label = response_scale.get(str(value))
        if not isinstance(label, str) or not label:
            raise ValueError(f"虚拟被试池缺少作答值 {value} 的文本标签")


def load_virtual_respondent_pool(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """读取并校验项目内置的匿名虚拟被试池。"""

    resolved_path = Path(path or DEFAULT_POOL_PATH).resolve()
    return deepcopy(_load_pool_cached(str(resolved_path)))


def build_virtual_pool_summary(
    pool: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """生成适合交互界面展示的被试池摘要。"""

    resolved_pool = dict(pool) if pool is not None else load_virtual_respondent_pool()
    items = resolved_pool["items"]
    facet_counts = Counter(item["facet_code"] for item in items)
    return {
        "pool_id": resolved_pool["pool_id"],
        "available_count": len(resolved_pool["respondents"]),
        "item_count": len(items),
        "facet_item_counts": dict(facet_counts),
        "source_file": resolved_pool["source"]["file_name"],
        "source_sha256": resolved_pool["source"]["source_sha256"],
        "contains_original_identifiers": False,
        "contains_demographics": False,
    }


def recommend_virtual_sample_size(available_count: int) -> int:
    """Recommend the minimum sample used by the automated development loop."""

    if (
        not isinstance(available_count, int)
        or isinstance(available_count, bool)
        or available_count < 1
    ):
        raise ValueError("available_count 必须是正整数")
    return min(available_count, MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE)


def build_virtual_sample_recommendations(
    available_count: int,
) -> list[dict[str, Any]]:
    """提供快速、平衡和全量三个不重复的样本量档位。"""

    recommended = recommend_virtual_sample_size(available_count)
    candidates = [
        (
            min(30, available_count),
            "quick",
            "最低开发样本",
            (
                "达到自动诊断筛选的最低人数；结果仍属于虚拟开发证据，"
                "不替代真实被试验证。"
            ),
        ),
        (
            min(100, available_count),
            "balanced",
            "平衡开发",
            "兼顾调用成本与初步分布检查，仍可能有明显抽样波动。",
        ),
        (
            available_count,
            "full",
            "全量样本",
            (
                "使用当前被试池的全部人格档案，避免额外的子样本选择波动；"
                "当前池容量不大于300人时推荐使用。"
            ),
        ),
    ]
    recommendations = []
    seen_sizes = set()
    for sample_size, code, label, description in candidates:
        if sample_size in seen_sizes:
            continue
        seen_sizes.add(sample_size)
        recommendations.append(
            {
                "code": code,
                "label": label,
                "sample_size": sample_size,
                "description": description,
                "recommended": sample_size == recommended,
            }
        )
    return recommendations


def select_virtual_respondent_refs(
    pool: Mapping[str, Any],
    sample_size: int,
    *,
    seed: int = DEFAULT_SELECTION_SEED,
) -> list[dict[str, Any]]:
    """按固定随机种子选择被试，仅把轻量引用写入工作流 State。"""

    respondents = pool.get("respondents")
    if not isinstance(respondents, list) or not respondents:
        raise ValueError("虚拟被试池为空")
    available_count = len(respondents)
    if (
        not isinstance(sample_size, int)
        or isinstance(sample_size, bool)
        or sample_size < 1
        or sample_size > available_count
    ):
        raise ValueError(
            f"sample_size 必须是 1 到 {available_count} 之间的整数"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed 必须是整数")

    if sample_size == available_count:
        selected_indexes = list(range(available_count))
    else:
        selected_indexes = random.Random(seed).sample(
            range(available_count),
            sample_size,
        )
    return [
        {
            "respondent_id": respondents[index]["respondent_id"],
            "pool_index": index,
        }
        for index in selected_indexes
    ]


def build_virtual_sample_config(
    pool: Mapping[str, Any],
    sample_size: int,
    *,
    seed: int = DEFAULT_SELECTION_SEED,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    max_retries: int = DEFAULT_MAX_RETRIES,
    persona_modes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """记录虚拟样本选择的可复现配置。"""

    summary = build_virtual_pool_summary(pool)
    if (
        not isinstance(max_concurrency, int)
        or isinstance(max_concurrency, bool)
        or max_concurrency < 1
        or max_concurrency > MAX_ALLOWED_CONCURRENCY
    ):
        raise ValueError(
            "max_concurrency 必须是 1 到 "
            f"{MAX_ALLOWED_CONCURRENCY} 之间的整数"
        )
    if (
        not isinstance(max_retries, int)
        or isinstance(max_retries, bool)
        or max_retries < 0
        or max_retries > 5
    ):
        raise ValueError("max_retries 必须是 0 到 5 之间的整数")
    resolved_persona_modes = list(
        DEFAULT_PERSONA_MODES if persona_modes is None else persona_modes
    )
    if not resolved_persona_modes:
        raise ValueError("persona_modes 不能为空")
    if (
        len(set(resolved_persona_modes)) != len(resolved_persona_modes)
        or any(
            mode not in SUPPORTED_PERSONA_MODES
            for mode in resolved_persona_modes
        )
    ):
        raise ValueError(
            "persona_modes 只能包含不重复的 summary_plus_items"
        )
    recommended = recommend_virtual_sample_size(summary["available_count"])
    return {
        "pool_id": summary["pool_id"],
        "pool_ref": str(DEFAULT_POOL_PATH),
        "source_file": summary["source_file"],
        "source_sha256": summary["source_sha256"],
        "available_count": summary["available_count"],
        "sample_size": sample_size,
        "recommended_sample_size": recommended,
        "automatic_selection_minimum_sample_size": (
            MINIMUM_AUTOMATIC_SELECTION_SAMPLE_SIZE
        ),
        "seed": seed,
        "max_concurrency": max_concurrency,
        "max_retries": max_retries,
        "persona_modes": resolved_persona_modes,
        "selection_strategy": (
            "all_in_source_order"
            if sample_size == summary["available_count"]
            else "simple_random_without_replacement"
        ),
        "persona_method": "Liu-SummaryPlusItems",
    }


def resolve_virtual_respondent_profiles(
    respondent_refs: Sequence[Mapping[str, Any]],
    pool: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """把 State 中的轻量引用解析为后续模型调用所需的人格档案。"""

    resolved_pool = pool or load_virtual_respondent_pool()
    respondents = resolved_pool["respondents"]
    items = resolved_pool["items"]
    response_scale = resolved_pool["response_scale"]
    profiles = []
    for respondent_ref in respondent_refs:
        pool_index = respondent_ref.get("pool_index")
        respondent_id = respondent_ref.get("respondent_id")
        if (
            not isinstance(pool_index, int)
            or isinstance(pool_index, bool)
            or pool_index < 0
            or pool_index >= len(respondents)
        ):
            raise ValueError("虚拟被试引用包含无效 pool_index")
        respondent = respondents[pool_index]
        if respondent["respondent_id"] != respondent_id:
            raise ValueError(
                f"虚拟被试引用 {respondent_id!r} 与被试池不一致"
            )
        personality_items = []
        for item, value in zip(items, respondent["response_values"]):
            personality_items.append(
                {
                    **item,
                    "response_value": value,
                    "response_label": response_scale[str(value)],
                }
            )
        profiles.append(
            {
                "respondent_id": respondent_id,
                "personality_items": personality_items,
            }
        )
    return profiles
