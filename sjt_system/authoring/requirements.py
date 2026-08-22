"""Minimal requirements clarification and confirmation rules."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re
from typing import Any

from sjt_system.authoring.construct_registry import (
    construct_selection_from_profile,
    resolve_construct_profile,
    resolve_construct_selection,
)
from sjt_system.config import DEFAULT_OUTPUT_LANGUAGE
from sjt_system.state import RequirementInteraction


USER_CONFIRMATION_FIELDS = frozenset(
    {"construct_selection", "target_population", "final_item_count"}
)
SPECIFICATION_FIELDS = frozenset(
    {*USER_CONFIRMATION_FIELDS, "output_language"}
)
VALID_SPECIFICATION_SOURCES = frozenset(
    {"user", "inferred", "system_default"}
)

_PLACEHOLDER_VALUES = {
    "n/a", "na", "none", "null", "unknown", "不明确", "不确定",
    "未知", "未指定", "待定", "待确认",
}
_LANGUAGE_ALIASES = {
    "简体中文": "zh-CN",
    "simplified chinese": "zh-CN",
    "zh_cn": "zh-CN",
    "zh-cn": "zh-CN",
    "中文": "zh-CN",
    "英文": "en",
    "english": "en",
}
_LANGUAGE_TAG = re.compile(r"^[a-z]{2,3}(?:-[A-Z]{2})?$")


def _is_valid_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().casefold()
    return bool(normalized) and normalized not in _PLACEHOLDER_VALUES


def _normalize_language(value: Any) -> str:
    if not _is_valid_text(value):
        return DEFAULT_OUTPUT_LANGUAGE
    text = str(value).strip()
    return _LANGUAGE_ALIASES.get(text.casefold(), text)


def _canonical_construct_selection(
    specification: Mapping[str, Any],
) -> dict[str, Any] | None:
    selection = specification.get("construct_selection")
    if isinstance(selection, Mapping):
        profile = resolve_construct_selection(selection)
        return construct_selection_from_profile(profile)

    # One-way compatibility for old requirement candidates/checkpoints. New
    # prompts never produce this overloaded free-text field.
    target = specification.get("target_construct")
    if isinstance(target, str) and target.strip():
        return construct_selection_from_profile(
            resolve_construct_profile(target)
        )
    return None


def _explicit_construct_selection(
    *texts: str | None,
) -> dict[str, Any] | None:
    """Resolve the latest unambiguous construct named by the user."""

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            profile = resolve_construct_profile(text)
        except ValueError:
            continue
        return construct_selection_from_profile(profile)
    return None


def canonicalize_requirement_agent_update(
    update: Mapping[str, Any],
    *,
    previous_candidate: Mapping[str, Any] | None = None,
    user_feedback: str | None = None,
    user_request: str | None = None,
) -> dict[str, Any]:
    """Normalize one model candidate into the four-field requirement spec."""

    del previous_candidate
    if set(update) != {"test_specification", "specification_sources"}:
        raise ValueError(
            "clarify_requirements 只能返回 test_specification 和 "
            "specification_sources"
        )
    raw_specification = update.get("test_specification")
    raw_sources = update.get("specification_sources")
    if not isinstance(raw_specification, Mapping):
        raise ValueError("test_specification 必须是对象")
    if not isinstance(raw_sources, Mapping):
        raise ValueError("specification_sources 必须是对象")

    explicit_selection = _explicit_construct_selection(
        user_feedback,
        user_request,
    )
    selection = (
        explicit_selection
        if explicit_selection is not None
        else _canonical_construct_selection(raw_specification)
    )
    specification = {
        "construct_selection": selection,
        "target_population": raw_specification.get("target_population"),
        "final_item_count": raw_specification.get("final_item_count"),
        "output_language": _normalize_language(
            raw_specification.get("output_language")
        ),
    }
    sources = {
        field: raw_sources.get(field)
        for field in SPECIFICATION_FIELDS
    }
    if sources.get("construct_selection") is None:
        sources["construct_selection"] = raw_sources.get("target_construct")
    if explicit_selection is not None:
        sources["construct_selection"] = "user"
    if sources.get("output_language") != "user":
        sources["output_language"] = "system_default"
    return {
        "test_specification": specification,
        "specification_sources": sources,
    }


def validate_test_specification(specification: Any) -> dict[str, str]:
    """Return field-level errors; an empty mapping means structurally valid."""

    if not isinstance(specification, Mapping):
        return {"test_specification": "必须是对象"}
    errors: dict[str, str] = {}
    if set(specification) != SPECIFICATION_FIELDS:
        errors["test_specification.fields"] = (
            "必须且只能包含 construct_selection、target_population、"
            "final_item_count、output_language"
        )
    try:
        resolve_construct_selection(specification.get("construct_selection"))
    except ValueError as exc:
        errors["construct_selection"] = str(exc)
    if not _is_valid_text(specification.get("target_population")):
        errors["target_population"] = "必须是非空且明确的文本"
    final_item_count = specification.get("final_item_count")
    if (
        isinstance(final_item_count, bool)
        or not isinstance(final_item_count, int)
        or final_item_count < 1
    ):
        errors["final_item_count"] = "必须是正整数"
    language = specification.get("output_language")
    if not isinstance(language, str) or not _LANGUAGE_TAG.fullmatch(language):
        errors["output_language"] = "必须是规范语言标签，例如 zh-CN 或 en"
    return errors


def build_requirement_interaction(
    result: Mapping[str, Any],
) -> RequirementInteraction:
    """Validate the only two pieces of interaction metadata still needed."""

    if not isinstance(result.get("suggestions"), list):
        raise ValueError("Requirement Agent 输出缺少 suggestions 列表")
    if not isinstance(result.get("questions"), list):
        raise ValueError("Requirement Agent 输出缺少 questions 列表")

    suggestions: list[dict[str, Any]] = []
    for suggestion in result["suggestions"]:
        if not isinstance(suggestion, Mapping):
            raise ValueError("Requirement suggestion 必须是对象")
        if set(suggestion) != {"field", "reason"}:
            raise ValueError("Requirement suggestion 只能包含 field 和 reason")
        field = suggestion.get("field")
        if field not in USER_CONFIRMATION_FIELDS:
            raise ValueError("Requirement suggestion 只能针对三个用户确认字段")
        if not _is_valid_text(suggestion.get("reason")):
            raise ValueError("Requirement suggestion 缺少有效 reason")
        suggestions.append(dict(suggestion))

    questions: list[dict[str, Any]] = []
    seen_fields: set[str] = set()
    for question in result["questions"]:
        if not isinstance(question, Mapping) or set(question) != {
            "field", "issue_type", "text"
        }:
            raise ValueError(
                "Requirement question 只能包含 field、issue_type、text"
            )
        field = question.get("field")
        if field not in USER_CONFIRMATION_FIELDS:
            raise ValueError("Requirement question 只能针对三个用户确认字段")
        if field in seen_fields:
            raise ValueError(f"Requirement question 重复字段：{field}")
        if question.get("issue_type") not in {
            "missing", "ambiguous", "confirm_inference"
        }:
            raise ValueError("Requirement question.issue_type 无效")
        if not _is_valid_text(question.get("text")):
            raise ValueError("Requirement question 缺少有效 text")
        seen_fields.add(str(field))
        questions.append(dict(question))
    if len(questions) > 3:
        raise ValueError("每轮最多提出三个需求问题")
    return {"suggestions": suggestions, "questions": questions}


def build_confirmed_requirement_fields_update(
    result: Mapping[str, Any],
    confirmed_fields: list[str],
) -> dict[str, Any]:
    """Merge explicit user-owned fields into durable confirmation state."""

    state_update = result.get("state_update")
    sources = (
        state_update.get("specification_sources", {})
        if isinstance(state_update, Mapping)
        else {}
    )
    user_fields = {
        field
        for field, source in sources.items()
        if source == "user" and field in USER_CONFIRMATION_FIELDS
    }
    return {
        "confirmed_requirement_fields": sorted(
            ({*confirmed_fields, *user_fields}) & USER_CONFIRMATION_FIELDS
        )
    }


def validate_requirement_confirmation(
    specification: Any,
    interaction: Mapping[str, Any] | None,
    confirmed_fields: list[str],
    specification_sources: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Validate structural completeness and explicit acceptance of inference."""

    errors = validate_test_specification(specification)
    if not isinstance(interaction, Mapping):
        return {**errors, "interaction": "缺少需求交互状态"}

    if not isinstance(specification_sources, Mapping):
        errors["specification_sources"] = "缺少字段来源记录"
        sources: Mapping[str, Any] = {}
    else:
        sources = specification_sources
        if set(sources) != SPECIFICATION_FIELDS:
            errors["specification_sources"] = "字段来源必须精确覆盖需求规格"
        invalid_sources = sorted(
            field for field, source in sources.items()
            if source not in VALID_SPECIFICATION_SOURCES
        )
        if invalid_sources:
            errors["invalid_specification_sources"] = (
                "以下字段来源无效：" + "、".join(invalid_sources)
            )
        for field in USER_CONFIRMATION_FIELDS:
            if sources.get(field) not in {"user", "inferred"}:
                errors[f"source.{field}"] = "来源必须是 user 或 inferred"
        if sources.get("output_language") not in {"user", "system_default"}:
            errors["source.output_language"] = (
                "输出语言来源必须是 user 或 system_default"
            )

    questions = interaction.get("questions")
    if not isinstance(questions, list):
        errors["questions"] = "必须是结构化问题列表"
    elif questions:
        errors["questions"] = "仍有待解决问题：" + "、".join(
            str(question.get("field"))
            for question in questions
            if isinstance(question, Mapping)
        )

    inferred = {
        field for field, source in sources.items()
        if source == "inferred" and field in USER_CONFIRMATION_FIELDS
    }
    unaccepted = sorted(inferred - set(confirmed_fields))
    if unaccepted:
        errors["inferred_fields"] = (
            "系统推断值尚未被用户接受：" + "、".join(unaccepted)
        )
    return errors
