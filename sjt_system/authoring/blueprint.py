"""Program-owned fixed blueprint validation and traversal."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sjt_system.authoring.generation_plan import (
    planned_generation_count,
    planned_retention_count,
    validate_generation_blueprint,
)


# Kept only for one-way checkpoint cleanup; current blueprints never contain
# these fields.
BLUEPRINT_REMOVED_FIELDS = {
    "option_count",
    "scoring_method",
    "target_final_item_count",
    "target_generation_item_count",
    "assumptions",
    "response_instruction_type",
    "response_format",
    "construct_model",
    "context_quotas",
    "item_specifications",
    "item_skeletons",
    "validation_summary",
}
CELL_REMOVED_FIELDS = {
    "behavioral_indicators",
    "context_type",
    "scoring_constraints",
    "allowed_context_categories",
}
def format_blueprint_errors_for_user(errors: Mapping[str, str]) -> str:
    return "细目表未通过程序校验：\n" + "\n".join(
        f"- {field}: {message}" for field, message in errors.items()
    )


def validate_blueprint_agent_update(update: Mapping[str, Any]) -> None:
    """The blueprint action may return only the fixed table."""

    allowed_fields = {"blueprint", "construct_profile"}
    if not set(update) <= allowed_fields or "blueprint" not in update:
        raise ValueError(
            "build_blueprint 只能提交 blueprint，以及可选的 construct_profile"
        )
    blueprint = update.get("blueprint")
    if not isinstance(blueprint, Mapping):
        raise ValueError("build_blueprint.blueprint 必须是对象")
    errors = validate_generation_blueprint(
        blueprint,
        {"final_item_count": planned_retention_count(blueprint)},
    )
    if errors:
        raise ValueError(format_blueprint_errors_for_user(errors))


def validate_integrated_blueprint(
    blueprint: Any,
    specification: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Validate only the current fixed GenerationBlueprint contract."""

    return validate_generation_blueprint(blueprint, specification)


def initialize_blueprint_progress(
    blueprint: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    """Initialize deterministic per-cell content progress."""

    return {
        str(cell["cell_id"]): {
            "generated": 0,
            "passed": 0,
            "rejected": 0,
            "missing": int(cell["planned_generation_count"]),
        }
        for cell in blueprint.get("cells") or []
        if isinstance(cell, Mapping)
    }


def select_next_blueprint_cell(
    blueprint: Mapping[str, Any],
    progress: Mapping[str, Mapping[str, int]],
) -> dict[str, Any] | None:
    """Return the first cell with an unattempted fixed slot."""

    for raw_cell in blueprint.get("cells") or []:
        if not isinstance(raw_cell, Mapping):
            continue
        cell = dict(raw_cell)
        cell_progress = progress.get(str(cell.get("cell_id"))) or {}
        if int(cell_progress.get("generated", 0)) < int(
            cell.get("planned_generation_count", 0)
        ):
            return cell
    return None


__all__ = [
    "BLUEPRINT_REMOVED_FIELDS",
    "CELL_REMOVED_FIELDS",
    "format_blueprint_errors_for_user",
    "initialize_blueprint_progress",
    "planned_generation_count",
    "planned_retention_count",
    "select_next_blueprint_cell",
    "validate_blueprint_agent_update",
    "validate_integrated_blueprint",
]
