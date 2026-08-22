"""Bounded model contexts, item slots, and cross-item diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any
from copy import deepcopy

from sjt_system.authoring.construct_registry import construct_selection_catalog


ITEM_MODEL_STATE_FIELDS = (
    "current_item",
    "current_item_specification",
    "current_blueprint_cell",
    "current_item_review",
    "test_specification",
    "user_feedback",
)


def build_item_model_state(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Project workflow state to the fields required by the item model."""

    projected = {
        field: deepcopy(state.get(field))
        for field in ITEM_MODEL_STATE_FIELDS
    }
    # Item/specification identity is workflow-owned routing state. Models see
    # the semantic payload only; option IDs remain because targeted option
    # repair needs them.
    if isinstance(projected.get("current_item"), dict):
        for field in ("item_id", "blueprint_cell_id"):
            projected["current_item"].pop(field, None)
    if isinstance(projected.get("current_item_specification"), dict):
        for field in (
            "specification_id",
            "blueprint_cell_id",
            "replacement_for_item_id",
        ):
            projected["current_item_specification"].pop(field, None)
    if isinstance(projected.get("current_blueprint_cell"), dict):
        projected["current_blueprint_cell"].pop("cell_id", None)
    cell = state.get("current_blueprint_cell")
    dimension_id = (
        cell.get("facet_id")
        if isinstance(cell, Mapping)
        else None
    )
    profile = state.get("construct_profile")
    projected["current_construct_dimension"] = None
    if isinstance(profile, Mapping):
        projected["construct_profile_ref"] = (
            (state.get("blueprint") or {}).get("construct_profile_ref")
            or state.get("construct_profile_ref")
        )
        projected["current_facet_profile"] = next(
            (
                dict(facet)
                for facet in profile.get("facets") or []
                if isinstance(facet, Mapping)
                and facet.get("facet_id") == dimension_id
            ),
            None,
        )
        projected["current_construct_dimension"] = projected[
            "current_facet_profile"
        ]
    return projected


def build_psychometric_repair_model_state(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Project only semantic item content to the psychometric repair agent."""

    projected = build_item_model_state(state)
    item = projected.get("current_item")
    if isinstance(item, Mapping):
        projected["current_item"] = {
            key: deepcopy(item[key])
            for key in ("scenario", "response_instruction", "response_options", "scoring_key")
            if item.get(key) is not None
        }
        projected["current_item"]["response_options"] = [
            {
                key: deepcopy(option[key])
                for key in ("option_id", "text", "behavioral_level")
                if option.get(key) is not None
            }
            for option in projected["current_item"].get("response_options") or []
            if isinstance(option, Mapping)
        ]
    for field in ("current_item_specification", "current_blueprint_cell"):
        value = projected.get(field)
        if isinstance(value, Mapping):
            projected[field] = _strip_repair_metadata(value)
    for field in ("current_facet_profile", "current_construct_dimension"):
        value = projected.get(field)
        if isinstance(value, Mapping):
            projected[field] = _strip_repair_metadata(value)
    projected.pop("construct_profile_ref", None)
    projected["test_specification"] = None
    projected["user_feedback"] = None
    projected["current_item_review"] = None
    return projected


def build_psychometric_repair_generation_context(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep only semantic generation context for a psychometric repair call."""

    context = build_item_generation_context(state)
    for field in ("current_item_specification", "current_blueprint_cell"):
        value = context.get(field)
        if isinstance(value, Mapping):
            context[field] = _strip_repair_metadata(value)
    context.pop("context_usage", None)
    context.pop("context_quotas", None)
    return context


def _strip_repair_metadata(value: Any) -> Any:
    """Recursively remove routing IDs while preserving the three useful refs."""

    if isinstance(value, Mapping):
        return {
            key: _strip_repair_metadata(raw)
            for key, raw in value.items()
            if key in {"option_id", "observation_id", "constraint_id"}
            or not str(key).endswith("_id")
        }
    if isinstance(value, list):
        return [_strip_repair_metadata(raw) for raw in value]
    return deepcopy(value)


def build_requirement_model_state(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Project state to the fields documented by the requirement prompt."""

    return {
        "user_request": state.get("user_request"),
        "test_specification": state.get("test_specification"),
        "pending_state_update": state.get("pending_state_update"),
        "specification_sources": state.get("specification_sources") or {},
        "user_feedback": state.get("user_feedback"),
        "confirmed_requirement_fields": (
            state.get("confirmed_requirement_fields") or []
        ),
        "requirement_conversation": (
            state.get("requirement_conversation") or []
        ),
        "construct_catalog": construct_selection_catalog(),
    }


def select_item_specification(
    specifications: list[dict[str, Any]],
    cell_id: str,
    generated_count: int,
) -> dict[str, Any] | None:
    """按该单元已经首次生成的题数选择下一个槽；重写不消耗新槽。"""

    candidates = [
        specification
        for specification in specifications
        if specification.get("blueprint_cell_id") == cell_id
    ]
    if generated_count < 0 or generated_count >= len(candidates):
        return None
    return dict(candidates[generated_count])


def _normalized(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text.lower())


def _ngrams(text: str, size: int = 2) -> set[str]:
    return {text[index:index + size] for index in range(max(0, len(text) - size + 1))}


def text_similarity(left: Any, right: Any) -> float:
    left_grams = _ngrams(_normalized(left))
    right_grams = _ngrams(_normalized(right))
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def build_item_pattern_profile(
    item: Mapping[str, Any],
    item_specification: Mapping[str, Any] | None,
) -> dict[str, Any]:
    options = item.get("response_options") or []
    scoring_key = item.get("scoring_key") or {}
    scored_options = [
        (option, scoring_key.get(option.get("option_id")))
        for option in options
        if isinstance(option, Mapping)
    ]
    valid_scored = [pair for pair in scored_options if isinstance(pair[1], (int, float))]
    highest = max(valid_scored, key=lambda pair: pair[1])[0].get("text", "") if valid_scored else ""
    lowest = min(valid_scored, key=lambda pair: pair[1])[0].get("text", "") if valid_scored else ""
    scenario = str(item.get("scenario", ""))
    return {
        "item_id": str(item.get("item_id", "")),
        "context_category": str(
            item.get("context_category")
            or (item_specification or {}).get("context_category", "未分类")
        ),
        "context_signature": str(item.get("context_signature") or _normalized(scenario)),
        "scenario": scenario,
        "highest_score_strategy": highest,
        "lowest_score_strategy": lowest,
        "option_lengths": [len(str(option.get("text", ""))) for option in options if isinstance(option, Mapping)],
        "score_position_pattern": [
            int(scoring_key[option.get("option_id")])
            for option in options
            if isinstance(option, Mapping)
            and isinstance(scoring_key.get(option.get("option_id")), (int, float))
        ],
    }


def build_deterministic_repair_tasks(
    item: Mapping[str, Any],
    item_specification: Mapping[str, Any] | None,
    profiles: Mapping[str, Mapping[str, Any]],
    blueprint_cell: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic answer-key-transparency checks."""

    tasks: list[dict[str, Any]] = []
    dimension_name = str((blueprint_cell or {}).get("facet_id") or "")
    dimension_terms = {
        term.strip()
        for term in re.split(r"[、,/与和及]", dimension_name)
        if len(term.strip()) >= 2
    }
    exposed_option_ids = sorted(
        {
            str(option.get("option_id"))
            for option in item.get("response_options") or []
            if isinstance(option, Mapping)
            and option.get("option_id")
            and any(
                term in str(option.get("text") or "")
                for term in dimension_terms
            )
        }
    )
    if exposed_option_ids:
        tasks.append(
            {
                "task_id": "deterministic-answer-key-transparency",
                "source": "deterministic",
                "targets": [
                    {
                        "field": "response_options",
                        "option_ids": exposed_option_ids,
                    }
                ],
                "problem": "反应选项直接使用目标构念标签，暴露评分方向",
                "instruction": "改用具体、可观察的行为描述",
            }
        )
    return tasks


def build_item_generation_context(state: Mapping[str, Any]) -> dict[str, Any]:
    item_specification = deepcopy(state.get("current_item_specification"))
    if isinstance(item_specification, dict):
        for field in (
            "specification_id",
            "blueprint_cell_id",
            "replacement_for_item_id",
        ):
            item_specification.pop(field, None)
    blueprint_cell = deepcopy(state.get("current_blueprint_cell"))
    if isinstance(blueprint_cell, dict):
        blueprint_cell.pop("cell_id", None)
    return {
        "current_item_specification": item_specification,
        "current_blueprint_cell": blueprint_cell,
        "context_usage": state.get("context_usage") or {},
        "context_quotas": (state.get("blueprint") or {}).get("context_quotas", []),
    }


def build_unified_review_context(state: Mapping[str, Any]) -> dict[str, Any]:
    """Provide bounded evidence for the four-dimension item diagnosis."""

    item = state.get("current_item") or {}
    cell = state.get("current_blueprint_cell") or {}
    dimension_id = cell.get("facet_id") or cell.get("dimension_id")
    profile = state.get("construct_profile") or {}
    dimension = next(
        (
            candidate
            for candidate in profile.get("facets") or []
            if isinstance(candidate, Mapping)
            and candidate.get("facet_id") == dimension_id
        ),
        None,
    )
    if dimension is None:
        raise ValueError(
            f"找不到当前蓝图单元对应的构念维度：{dimension_id!r}"
        )
    review_item = deepcopy(dict(item))
    review_item.pop("item_id", None)
    review_item.pop("blueprint_cell_id", None)
    review_cell = deepcopy(dict(cell))
    review_cell.pop("cell_id", None)
    review_specification = deepcopy(state.get("current_item_specification"))
    if isinstance(review_specification, dict):
        for field in (
            "specification_id",
            "blueprint_cell_id",
            "replacement_for_item_id",
        ):
            review_specification.pop(field, None)
    return {
        "current_item": review_item,
        "construct_dimension": dict(dimension),
        "behavioral_anchors": (
            (state.get("current_item_specification") or {}).get(
                "behavioral_anchors"
            )
        ),
        "test_specification": state.get("test_specification"),
        "blueprint_cell": review_cell,
        "item_specification": review_specification,
    }
