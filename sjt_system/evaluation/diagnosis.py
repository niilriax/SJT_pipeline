"""Construct-constrained diagnosis support for post-simulation item repair.

The functions in this module deliberately separate three responsibilities:

* deterministic statistics decide whether an item needs diagnosis;
* the evidence packet exposes the normal construct model and observations;
* validators restrict an LLM diagnosis and patch to one evidence-backed edit.

They do not infer a textual defect from a statistical threshold.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any, Mapping


CITC_DIAGNOSIS_THRESHOLD = 0.20
DIFFICULTY_LOWER_BOUND = 0.20
DIFFICULTY_UPPER_BOUND = 0.80
MINIMUM_EFFECTIVE_OPTION_COUNT = 3

_EDITABLE_COMPONENTS = {"scenario", "response_options"}
_UPSTREAM_COMPONENTS = {
    "skeleton",
    "activation_mechanism",
    "behavior_evidence",
    "construct",
}
_NON_ACTIONABLE_COMPONENTS = {
    "simulation",
    "simulation_or_insufficient_evidence",
    "insufficient_evidence",
}
_ALL_COMPONENTS = (
    _EDITABLE_COMPONENTS | _UPSTREAM_COMPONENTS | _NON_ACTIONABLE_COMPONENTS
)
_LEVEL_RANK = {"low": 0, "medium_low": 1, "medium_high": 2, "high": 3}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _citc_value(statistics: Mapping[str, Any]) -> float | None:
    payload = (
        statistics.get("facet_corrected_item_total_correlation")
        or statistics.get("corrected_item_total_correlation")
        or {}
    )
    return _number(payload.get("r")) if isinstance(payload, Mapping) else None


def _effective_option_count(statistics: Mapping[str, Any]) -> int | None:
    quality = statistics.get("quality_evaluation") or {}
    value = quality.get("effective_option_count") if isinstance(quality, Mapping) else None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def item_requires_psychometric_diagnosis(statistics: Mapping[str, Any]) -> bool:
    """Return whether current deterministic evidence triggers LLM diagnosis."""

    citc = _citc_value(statistics)
    difficulty = _number(statistics.get("difficulty"))
    effective_options = _effective_option_count(statistics)
    return (
        citc is None
        or citc < CITC_DIAGNOSIS_THRESHOLD
        or difficulty is None
        or difficulty < DIFFICULTY_LOWER_BOUND
        or difficulty > DIFFICULTY_UPPER_BOUND
        or effective_options is None
        or effective_options < MINIMUM_EFFECTIVE_OPTION_COUNT
    )


def _find_item(state: Mapping[str, Any], item_id: str) -> dict[str, Any]:
    pools = [
        state.get("frozen_item_bank") or [],
        state.get("item_pool") or [],
        state.get("selected_items") or [],
    ]
    for pool in pools:
        for candidate in pool:
            if isinstance(candidate, Mapping) and str(candidate.get("item_id")) == item_id:
                return deepcopy(dict(candidate))
    raise ValueError(f"item_id cannot be resolved: {item_id}")


def _find_specification(state: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    item_id = str(item.get("item_id") or "")
    cell_id = str(item.get("blueprint_cell_id") or "")
    for candidate in state.get("item_specifications") or []:
        if not isinstance(candidate, Mapping):
            continue
        if (
            str(candidate.get("specification_id") or "") == item_id
            or str(candidate.get("item_id") or "") == item_id
        ):
            return deepcopy(dict(candidate))
    blueprint = state.get("blueprint") or {}
    for slot in blueprint.get("slots") or []:
        if not isinstance(slot, Mapping):
            continue
        if str(slot.get("specification_id") or "") != item_id:
            continue
        for candidate in state.get("item_specifications") or []:
            if (
                isinstance(candidate, Mapping)
                and str(candidate.get("specification_id") or "")
                == str(slot.get("specification_id") or "")
            ):
                return deepcopy(dict(candidate))
    if cell_id:
        for candidate in state.get("item_specifications") or []:
            if (
                isinstance(candidate, Mapping)
                and str(candidate.get("blueprint_cell_id") or "") == cell_id
            ):
                return deepcopy(dict(candidate))
    raise ValueError(f"item specification cannot be resolved: {item_id}")


def _find_cell(state: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    cell_id = str(item.get("blueprint_cell_id") or "")
    for candidate in (state.get("blueprint") or {}).get("cells") or []:
        if isinstance(candidate, Mapping) and str(candidate.get("cell_id")) == cell_id:
            return deepcopy(dict(candidate))
    raise ValueError(f"blueprint cell cannot be resolved: {cell_id}")


def _find_facet_and_behavior(
    state: Mapping[str, Any],
    item: Mapping[str, Any],
    cell: Mapping[str, Any],
    specification: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    facet_id = _text(
        cell.get("facet_id")
        or cell.get("dimension_id")
        or item.get("target_dimension_id")
    )
    behavior_id = _text(
        cell.get("behavior_id")
        or cell.get("behavior_evidence_id")
        or specification.get("behavior_evidence_id")
    )
    profile = (state.get("blueprint") or {}).get("construct_profile_snapshot") or {}
    facet = next(
        (
            deepcopy(dict(candidate))
            for candidate in profile.get("facets") or []
            if isinstance(candidate, Mapping)
            and str(candidate.get("facet_id") or "") == facet_id
        ),
        None,
    )
    if facet is None:
        raise ValueError(f"facet cannot be resolved: {facet_id}")
    behavior = next(
        (
            deepcopy(dict(candidate))
            for candidate in facet.get("behavior_evidence") or []
            if isinstance(candidate, Mapping)
            and str(candidate.get("behavior_id") or "") == behavior_id
        ),
        None,
    )
    if behavior is None:
        raise ValueError(f"behavior evidence cannot be resolved: {behavior_id}")
    return facet, behavior


def _find_skeleton(state: Mapping[str, Any], specification: Mapping[str, Any]) -> dict[str, Any]:
    specification_id = _text(specification.get("specification_id"))
    candidates = state.get("item_skeletons") or {}
    if isinstance(candidates, Mapping):
        value = candidates.get(specification_id)
        if isinstance(value, Mapping):
            return deepcopy(dict(value))
    raise ValueError(f"item skeleton cannot be resolved: {specification_id}")


def _append_constraint(
    rows: list[dict[str, Any]],
    constraint_id: str,
    component: str,
    statement: Any,
) -> None:
    text = _text(statement)
    if text:
        rows.append(
            {
                "constraint_id": constraint_id,
                "component": component,
                "statement": text,
            }
        )


def _option_rates(statistics: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, float]:
    direct = statistics.get("option_selection_rates")
    if isinstance(direct, Mapping):
        return {
            str(option_id): float(rate)
            for option_id, rate in direct.items()
            if _number(rate) is not None
        }
    rows = statistics.get("option_statistics") or {}
    if isinstance(rows, Mapping):
        return {
            str(option_id): float(
                row.get("selection_rate", row.get("rate"))
            )
            for option_id, row in rows.items()
            if isinstance(row, Mapping)
            and _number(row.get("selection_rate", row.get("rate"))) is not None
        }
    return {
        str(option.get("option_id")): 0.0
        for option in item.get("response_options") or []
        if isinstance(option, Mapping) and option.get("option_id")
    }


def _latest_content_review(state: Mapping[str, Any], item_id: str) -> dict[str, Any] | None:
    direct = (state.get("item_reviews") or {}).get(item_id)
    if isinstance(direct, Mapping):
        return deepcopy(dict(direct))
    if isinstance(direct, list) and direct:
        for candidate in reversed(direct):
            if isinstance(candidate, Mapping):
                return deepcopy(dict(candidate))
    history = (state.get("item_history") or {}).get(item_id) or []
    for candidate in reversed(history):
        if isinstance(candidate, Mapping) and isinstance(candidate.get("review"), Mapping):
            return deepcopy(dict(candidate["review"]))
    return None


def build_construct_diagnosis_evidence(
    state: Mapping[str, Any],
    item_id: str,
    *,
    revision_round: int,
) -> dict[str, Any]:
    """Resolve a diagnosis packet from stable workflow references."""

    item = _find_item(state, item_id)
    specification = _find_specification(state, item)
    cell = _find_cell(state, item)
    facet, behavior = _find_facet_and_behavior(state, item, cell, specification)
    skeleton = _find_skeleton(state, specification)

    behavior_id = _text(behavior.get("behavior_id"))
    mechanism_id = _text(cell.get("mechanism_id") or specification.get("mechanism_id"))
    situation_id = _text(cell.get("situation_id") or specification.get("situation_id"))
    constraints: list[dict[str, Any]] = []
    _append_constraint(
        constraints,
        f"FACET_DEFINITION:{facet.get('facet_id')}",
        "facet",
        facet.get("definition"),
    )
    _append_constraint(constraints, f"BE_DEFINITION:{behavior_id}", "behavior_evidence", behavior.get("observable_behavior"))
    _append_constraint(constraints, f"BE_HIGH:{behavior_id}", "behavior_evidence", behavior.get("high_expression"))
    _append_constraint(constraints, f"BE_LOW:{behavior_id}", "behavior_evidence", behavior.get("low_expression"))
    _append_constraint(constraints, f"BE_BOUNDARY:{behavior_id}", "behavior_evidence", behavior.get("boundary_condition"))
    for source_field, label in (
        ("common_confounds", "FACET_CONFOUND"),
        ("inappropriate_contexts", "FACET_INAPPROPRIATE"),
        ("inappropriate_conditions", "FACET_INAPPROPRIATE"),
        ("forbidden_patterns", "FACET_FORBIDDEN"),
        ("option_design_rules", "FACET_OPTION_RULE"),
    ):
        for index, statement in enumerate(facet.get(source_field) or [], start=1):
            _append_constraint(constraints, f"{label}:{index}", "facet", statement)
    _append_constraint(
        constraints,
        f"MECHANISM:{mechanism_id}",
        "activation_mechanism",
        cell.get("activation_mechanism") or specification.get("activation_mechanism"),
    )
    situation_statement = " | ".join(
        filter(
            None,
            [
                _text(cell.get("domain") or specification.get("context_category")),
                _text(cell.get("actor_relation") or specification.get("social_context")),
                _text(cell.get("event_class") or specification.get("context_seed")),
            ],
        )
    )
    _append_constraint(constraints, f"SITUATION:{situation_id}", "situation", situation_statement)
    _append_constraint(
        constraints,
        f"SKELETON:TENSION:{specification.get('specification_id')}",
        "skeleton",
        skeleton.get("behavioral_tension") or specification.get("core_tension") or specification.get("core_behavioral_tension"),
    )
    option_structure = skeleton.get("option_structure") or specification.get("behavioral_anchors") or {}
    if isinstance(option_structure, Mapping):
        option_rows = [
            {"behavioral_level": level, "behavioral_tendency": statement}
            for level, statement in option_structure.items()
        ]
    else:
        option_rows = option_structure
    for row in option_rows or []:
        if not isinstance(row, Mapping):
            continue
        level = _text(row.get("behavioral_level"))
        statement = row.get("behavioral_tendency") or row.get("text")
        _append_constraint(constraints, f"SKELETON:OPTION:{level}", "skeleton", statement)

    statistics = deepcopy(dict((state.get("item_statistics") or {}).get(item_id) or {}))
    rates = _option_rates(statistics, item)
    observations = [
        {"observation_id": "OBS:CITC", "metric": "facet_citc", "value": _citc_value(statistics)},
        {"observation_id": "OBS:DIFFICULTY", "metric": "standardized_mean_score", "value": _number(statistics.get("difficulty"))},
        {"observation_id": "OBS:EFFECTIVE_OPTION_COUNT", "metric": "effective_option_count", "value": _effective_option_count(statistics)},
    ]
    observations.extend(
        {
            "observation_id": f"OBS:OPTION_RATE:{option_id}",
            "metric": "option_selection_rate",
            "option_id": option_id,
            "value": rate,
        }
        for option_id, rate in sorted(rates.items())
    )
    option_evidence = []
    for option in item.get("response_options") or []:
        if not isinstance(option, Mapping):
            continue
        option_id = _text(option.get("option_id"))
        option_evidence.append(
            {
                "option_id": option_id,
                "text": option.get("text"),
                "behavioral_level": option.get("behavioral_level"),
                "score": (item.get("scoring_key") or {}).get(option_id),
                "selection_rate": rates.get(option_id),
            }
        )
    prior_repairs = [
        deepcopy(dict(row))
        for row in state.get("psychometric_repair_history") or []
        if (
            isinstance(row, Mapping)
            and str(row.get("item_id")) == item_id
            and row.get("event") == "psychometric_item_repaired"
        )
    ]
    return {
        "item_id": item_id,
        "item_version": item.get("version"),
        "revision_round": revision_round,
        "blueprint_refs": {
            "blueprint_cell_id": item.get("blueprint_cell_id"),
            "facet_id": facet.get("facet_id"),
            "behavior_evidence_id": behavior_id,
            "mechanism_id": mechanism_id,
            "situation_id": situation_id,
            "specification_id": specification.get("specification_id"),
        },
        "normal_constraints": constraints,
        "observations": observations,
        "current_item": item,
        "option_evidence": option_evidence,
        "latest_content_review": _latest_content_review(state, item_id),
        "prior_atomic_repairs": prior_repairs,
    }


def diagnosis_fingerprint(evidence: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(evidence))
    payload.pop("diagnosis_fingerprint", None)
    payload.pop("revision_round", None)
    payload.pop("prior_atomic_repairs", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _validate_refs(
    refs: Any,
    valid: set[str],
    *,
    label: str,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(refs, list) or not all(isinstance(value, str) for value in refs):
        raise ValueError(f"{label} refs must be a list of strings")
    if not refs and not allow_empty:
        raise ValueError(f"{label} refs cannot be empty")
    unknown = set(refs) - valid
    if unknown:
        raise ValueError(f"unknown {label} reference: {sorted(unknown)}")
    return refs


def _validate_atomic_edit(
    atomic_edit: Any,
    *,
    selected: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    """Validate one edit while keeping the expert level and scoring immutable."""

    if not isinstance(atomic_edit, Mapping) or set(atomic_edit) != {
        "target_field", "option_ids", "problem", "instruction"
    }:
        raise ValueError("repair task requires one atomic_edit")
    components = selected.get("suspect_components") or []
    if len(components) != 1 or components[0] not in _EDITABLE_COMPONENTS:
        raise ValueError("repair task must point to one editable component")
    target_field = atomic_edit.get("target_field")
    if target_field != components[0]:
        raise ValueError("atomic edit target must match selected diagnosis")
    option_ids = atomic_edit.get("option_ids")
    if not isinstance(option_ids, list) or not all(
        isinstance(value, str) for value in option_ids
    ):
        raise ValueError("atomic edit option_ids are invalid")
    valid_option_ids = {
        str(row.get("option_id"))
        for row in evidence.get("option_evidence") or []
        if isinstance(row, Mapping)
    }
    if target_field == "scenario" and option_ids:
        raise ValueError("scenario edit cannot include option_ids")
    if target_field == "scenario" and selected.get("affected_option_ids"):
        raise ValueError("scenario diagnosis cannot include affected_option_ids")
    if target_field == "response_options":
        if not 1 <= len(option_ids) <= 2 or not set(option_ids).issubset(valid_option_ids):
            raise ValueError("response_options edit must name 1-2 valid options")
        if set(option_ids) != set(selected.get("affected_option_ids") or []):
            raise ValueError("atomic edit options must match selected diagnosis")
        if len(option_ids) == 2:
            levels = {
                str(row.get("option_id")): str(row.get("behavioral_level"))
                for row in evidence.get("option_evidence") or []
                if isinstance(row, Mapping)
            }
            ranks = [_LEVEL_RANK.get(levels.get(option_id, "")) for option_id in option_ids]
            if None in ranks or abs(ranks[0] - ranks[1]) != 1:
                raise ValueError("two option edits must target adjacent behavioral levels")
    if not _text(atomic_edit.get("problem")) or not _text(atomic_edit.get("instruction")):
        raise ValueError("atomic edit problem and instruction cannot be empty")


def validate_atomic_repair_advice(advice: Any, evidence: Mapping[str, Any]) -> None:
    """Validate all confirmed, evidence-backed edits for one item.

    ``repair_tasks`` is the current contract.  The former
    ``selected_diagnosis_id``/``atomic_edit`` pair remains accepted so older
    checkpoints and test fixtures can be resumed safely.
    """

    expected = {
        "item_id",
        "decision",
        "observed_discrepancies",
        "candidate_diagnoses",
        "summary",
    }
    legacy_fields = {"selected_diagnosis_id", "atomic_edit"}
    current_fields = {"repair_tasks"}
    if not isinstance(advice, dict) or not (
        set(advice) == expected | legacy_fields
        or set(advice) == expected | current_fields
        or set(advice) == expected | legacy_fields | current_fields
    ):
        raise ValueError("AtomicRepairAdvice fields are invalid")
    if str(advice.get("item_id")) != str(evidence.get("item_id")):
        raise ValueError("AtomicRepairAdvice item_id does not match evidence")
    decision = advice.get("decision")
    if decision not in {"repair", "defer"}:
        raise ValueError("AtomicRepairAdvice decision must be repair or defer")
    valid_observations = {
        str(row.get("observation_id"))
        for row in evidence.get("observations") or []
        if isinstance(row, Mapping)
    }
    valid_constraints = {
        str(row.get("constraint_id"))
        for row in evidence.get("normal_constraints") or []
        if isinstance(row, Mapping)
    }
    valid_option_ids = {
        str(row.get("option_id"))
        for row in evidence.get("option_evidence") or []
        if isinstance(row, Mapping)
    }
    discrepancies = advice.get("observed_discrepancies")
    if not isinstance(discrepancies, list) or not discrepancies:
        raise ValueError("observed_discrepancies cannot be empty")
    for row in discrepancies:
        if not isinstance(row, Mapping) or set(row) != {
            "observation_refs", "constraint_refs", "description"
        }:
            raise ValueError("observed discrepancy fields are invalid")
        _validate_refs(row.get("observation_refs"), valid_observations, label="observation")
        _validate_refs(row.get("constraint_refs"), valid_constraints, label="constraint", allow_empty=True)
        if not _text(row.get("description")):
            raise ValueError("observed discrepancy description cannot be empty")
    candidates = advice.get("candidate_diagnoses")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
        raise ValueError("candidate_diagnoses must contain 1-3 diagnoses")
    candidate_ids: set[str] = set()
    for row in candidates:
        if not isinstance(row, Mapping) or set(row) != {
            "diagnosis_id", "suspect_components", "affected_option_ids",
            "observation_refs", "constraint_refs", "textual_evidence",
            "explanation", "confidence",
        }:
            raise ValueError("candidate diagnosis fields are invalid")
        diagnosis_id = _text(row.get("diagnosis_id"))
        if not diagnosis_id or diagnosis_id in candidate_ids:
            raise ValueError("diagnosis_id must be non-empty and unique")
        candidate_ids.add(diagnosis_id)
        components = row.get("suspect_components")
        if (
            not isinstance(components, list)
            or not components
            or not all(component in _ALL_COMPONENTS for component in components)
        ):
            raise ValueError("suspect_components contain an invalid component")
        option_ids = row.get("affected_option_ids")
        if not isinstance(option_ids, list) or not all(isinstance(value, str) for value in option_ids):
            raise ValueError("affected_option_ids must be a list of strings")
        if not set(option_ids).issubset(valid_option_ids):
            raise ValueError("candidate diagnosis points to an unknown option")
        if "response_options" not in components and option_ids:
            raise ValueError("non-option diagnosis cannot include affected_option_ids")
        _validate_refs(row.get("observation_refs"), valid_observations, label="observation")
        _validate_refs(row.get("constraint_refs"), valid_constraints, label="constraint")
        if row.get("confidence") not in {"low", "medium", "high"}:
            raise ValueError("candidate confidence is invalid")
        if not _text(row.get("explanation")):
            raise ValueError("candidate explanation cannot be empty")
    repair_tasks = advice.get("repair_tasks")
    if repair_tasks is None:
        selected_id = advice.get("selected_diagnosis_id")
        atomic_edit = advice.get("atomic_edit")
        if decision == "defer":
            if selected_id is not None or atomic_edit is not None:
                raise ValueError("defer cannot select a diagnosis or atomic edit")
            return
        if not isinstance(selected_id, str) or selected_id not in candidate_ids:
            raise ValueError("repair must select a candidate diagnosis")
        selected = next(row for row in candidates if row.get("diagnosis_id") == selected_id)
        if selected.get("confidence") not in {"medium", "high"}:
            raise ValueError("selected diagnosis confidence must be medium or high")
        if not _text(selected.get("textual_evidence")):
            raise ValueError("selected repair diagnosis requires concrete textual evidence")
        _validate_atomic_edit(atomic_edit, selected=selected, evidence=evidence)
        return

    if not isinstance(repair_tasks, list) or len(repair_tasks) > 4:
        raise ValueError("repair_tasks must contain 0-4 tasks")
    if decision == "defer" and repair_tasks:
        raise ValueError("defer cannot contain repair_tasks")
    if decision == "repair" and not repair_tasks:
        raise ValueError("repair requires at least one repair_task")
    seen_scopes: set[tuple[str, tuple[str, ...]]] = set()
    seen_option_ids: set[str] = set()
    for task in repair_tasks:
        if not isinstance(task, Mapping) or set(task) != {"diagnosis_id", "atomic_edit"}:
            raise ValueError("repair task fields are invalid")
        diagnosis_id = _text(task.get("diagnosis_id"))
        if diagnosis_id not in candidate_ids:
            raise ValueError("repair task references unknown diagnosis")
        selected = next(row for row in candidates if row.get("diagnosis_id") == diagnosis_id)
        if selected.get("confidence") not in {"medium", "high"}:
            raise ValueError("repair task confidence must be medium or high")
        if not _text(selected.get("textual_evidence")):
            raise ValueError("repair task requires concrete textual evidence")
        atomic_edit = task.get("atomic_edit")
        _validate_atomic_edit(atomic_edit, selected=selected, evidence=evidence)
        scope = (
            str(atomic_edit.get("target_field")),
            tuple(sorted(str(option_id) for option_id in atomic_edit.get("option_ids") or [])),
        )
        if scope in seen_scopes:
            raise ValueError("repair_tasks contain duplicate edit scopes")
        seen_scopes.add(scope)
        if atomic_edit.get("target_field") == "response_options":
            option_ids = set(atomic_edit.get("option_ids") or [])
            if seen_option_ids & option_ids:
                raise ValueError("repair_tasks contain overlapping option edit scopes")
            seen_option_ids.update(option_ids)


def repair_tasks_from_advice(advice: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the confirmed edit tasks, including legacy single-edit advice."""

    tasks = advice.get("repair_tasks")
    if isinstance(tasks, list):
        return [deepcopy(dict(task)) for task in tasks if isinstance(task, Mapping)]
    selected_id = advice.get("selected_diagnosis_id")
    atomic_edit = advice.get("atomic_edit")
    if isinstance(selected_id, str) and isinstance(atomic_edit, Mapping):
        return [
            {
                "diagnosis_id": selected_id,
                "atomic_edit": deepcopy(dict(atomic_edit)),
            }
        ]
    return []


def validate_atomic_item_patch(
    patch: Any,
    current_item: Mapping[str, Any],
    advice: Mapping[str, Any],
) -> None:
    """Reject a patch that changes anything outside the selected atomic scope."""

    required = {"scenario_update", "option_updates"}
    allowed = {*required, "change_summary"}
    if (
        not isinstance(patch, Mapping)
        or not required.issubset(patch)
        or not set(patch).issubset(allowed)
    ):
        raise ValueError("item patch fields are invalid")
    atomic_edit = advice.get("atomic_edit") or {}
    target_field = atomic_edit.get("target_field")
    scenario_update = patch.get("scenario_update")
    option_updates = patch.get("option_updates")
    if not isinstance(option_updates, list):
        raise ValueError("option_updates must be a list")
    if scenario_update is not None and option_updates:
        raise ValueError("an atomic patch may change only scenario or options, not both")
    if target_field == "scenario":
        if not isinstance(scenario_update, str) or not scenario_update.strip():
            raise ValueError("scenario patch must change the scenario")
        if scenario_update.strip() == _text(current_item.get("scenario")):
            raise ValueError("scenario patch did not change the selected field")
        if option_updates:
            raise ValueError("scenario patch may change only the scenario")
        return
    if target_field != "response_options":
        raise ValueError("atomic patch target is not editable")
    if scenario_update is not None:
        raise ValueError("option patch may change only selected options")
    expected_ids = set(atomic_edit.get("option_ids") or [])
    update_ids = {
        str(row.get("option_id"))
        for row in option_updates
        if isinstance(row, Mapping)
    }
    if update_ids != expected_ids or len(option_updates) != len(expected_ids):
        raise ValueError("option patch must change only all selected options")
    current = {
        str(row.get("option_id")): _text(row.get("text"))
        for row in current_item.get("response_options") or []
        if isinstance(row, Mapping)
    }
    for row in option_updates:
        if not isinstance(row, Mapping) or set(row) != {"option_id", "text"}:
            raise ValueError("option update fields are invalid")
        option_id = str(row.get("option_id"))
        new_text = _text(row.get("text"))
        if not new_text or new_text == current.get(option_id):
            raise ValueError("option patch did not change every selected option")
