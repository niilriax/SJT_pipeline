"""Dispatch LLM-backed and deterministic workflow actions."""

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from langchain_core.runnables import Runnable

from sjt_system.agent.agent_factory import (
    PSYCHOMETRIC_REASONING_ROLE_MANIFEST,
    compact_skeleton_agent,
    item_regeneration_agent,
    item_review_agent,
    item_writer_agent,
    psychometric_item_repair_agent,
    psychometric_repair_diagnosis_agent,
    requirement_agent,
    revision_agent,
)
from sjt_system.authoring.blueprint import (
    format_blueprint_errors_for_user,
)
from sjt_system.authoring.construct_registry import resolve_specification_profile
from sjt_system.authoring.generation_plan import (
    build_generation_blueprint,
    classify_compact_skeletons,
    materialize_item_specifications,
    planned_generation_count,
    validate_generation_blueprint,
    required_generation_total,
    resolve_blueprint_design,
)
from sjt_system.state import PSJTRouteDecision, PSJTState
from sjt_system.authoring.context import (
    build_item_generation_context,
    build_item_model_state,
    build_requirement_model_state,
    build_unified_review_context,
)
from sjt_system.authoring.items import (
    canonicalize_item_agent_update,
    validate_item_agent_update,
    validate_item_review,
    validate_item_review_diagnosis,
)
from sjt_system.authoring.bank import build_item_bank_freeze_update
from sjt_system.agent.retry import ainvoke_model_with_schema_repair
from sjt_system.runtime.progress import emit_progress
from sjt_system.evaluation.simulation import run_virtual_response_simulation
from sjt_system.evaluation.psychometrics import run_psychometric_analysis
from sjt_system.evaluation.selection import (
    build_psychometric_repair_evidence,
    _psychometric_repair_entry,
    psychometric_diagnosis_to_review,
    run_item_selection,
    validate_psychometric_repair_diagnosis,
)
from sjt_system.evaluation.diagnosis import (
    build_construct_diagnosis_evidence,
    diagnosis_fingerprint,
    item_requires_psychometric_diagnosis,
    repair_tasks_from_advice,
    validate_atomic_item_patch,
    validate_atomic_repair_advice,
)
from sjt_system.knowledge.behavior_evidence import (
    attach_behavior_evidence,
    load_ipip_corpus,
)
from sjt_system.knowledge.behavior_evidence_agents import ensure_behavior_evidence
from sjt_system.authoring.situation_space import (
    ensure_facet_expansion,
    propose_blueprint_rows,
)
from sjt_system.delivery.assembly import run_test_assembly
from sjt_system.delivery.lifecycle import run_test_rescore, run_test_review
from sjt_system.delivery.reporting import run_report_generation


MAX_ITEM_OUTPUT_CANDIDATES = 2
MAX_ITEM_SKELETON_ATTEMPTS = 1


def distribute_situation_quotas(
    final_item_count: int, facet_count: int
) -> list[int]:
    """Distribute the requested unique situations evenly across facets."""

    final_item_count = int(final_item_count)
    facet_count = int(facet_count)
    if final_item_count < 1 or facet_count < 1:
        raise ValueError("题数和 facet 数必须为正整数")
    if final_item_count < facet_count:
        raise ValueError("最终题数必须不少于所选 facet 数")
    base, remainder = divmod(final_item_count, facet_count)
    return [base + (1 if index < remainder else 0) for index in range(facet_count)]


class SkeletonDevelopmentFailure(ValueError):
    """One fixed slot exhausted its bounded skeleton-development budget."""

    def __init__(
        self,
        specification_id: str,
        history: list[dict[str, Any]],
        reason: str | None = None,
    ) -> None:
        self.specification_id = specification_id
        self.history = deepcopy(history)
        super().__init__(
            "当前固定槽位的心理骨架未通过程序确定性校验"
            + (f"：{reason}" if reason else "")
        )


class ItemReviewProcessError(ValueError):
    """The reviewer exhausted retries without producing a valid review."""

    def __init__(
        self,
        message: str,
        *,
        item_id: str,
        item_version: int,
        review_request_id: str,
        attempt_count: int,
    ) -> None:
        super().__init__(message)
        self.item_id = item_id
        self.item_version = item_version
        self.review_request_id = review_request_id
        self.attempt_count = attempt_count


# 这里只注册 LLM 任务；统计、筛选、组卷和重计分后续注册确定性函数。
AGENT_MAP: dict[str, Runnable] = {
    "clarify_requirements": requirement_agent,
    "generate_item": item_writer_agent,
    "revise_item": revision_agent,
    "regenerate_item": item_regeneration_agent,
}


async def _ainvoke_model(
    agent: Runnable,
    input_data: dict[str, Any],
    *,
    job_label: str,
) -> Any:
    """Invoke one model request with a bounded end-to-end timeout."""

    return await ainvoke_model_with_schema_repair(
        agent,
        input_data,
        job_label=job_label,
        max_schema_repair_attempts=0,
    )


def _output_error_kind(exc: ValueError) -> str:
    """Classify output failures without conflating JSON and business errors."""

    value = getattr(exc, "error_kind", None)
    return value if isinstance(value, str) else "business_validation"


def _invalid_candidate(
    result: Any,
    exc: ValueError,
    *,
    state_update_only: bool = False,
) -> Any:
    """Preserve parser/schema candidates that failed before result assignment."""

    candidate = getattr(exc, "candidate", None)
    if candidate is not None:
        return candidate
    if state_update_only and isinstance(result, dict):
        return result.get("state_update")
    return result


async def execute_item_review(state: PSJTState) -> dict[str, Any]:
    """Run one diagnosis call, then derive all repair tasks in code."""

    current_item = state.get("current_item")
    if not isinstance(current_item, Mapping):
        raise ValueError("review_item requires current_item")
    item_id = current_item.get("item_id")
    item_version = current_item.get("version")
    if not isinstance(item_id, str) or not item_id.strip():
        raise ValueError("review_item current_item is missing item_id")
    if not isinstance(item_version, int) or isinstance(item_version, bool):
        raise ValueError("review_item current_item is missing item version")
    review_request_id = (
        f"{state.get('run_id')}:{item_id}:v{item_version}:"
        f"review:{int(state.get('step_count') or 0)}"
    )
    context = build_unified_review_context(state)
    input_data = context
    last_error: ValueError | None = None
    max_attempts = 4
    for attempt in range(max_attempts):
        result: Any = None
        try:
            result = await _ainvoke_model(
                item_review_agent,
                {"input_data": input_data},
                job_label="统一题目审查",
            )
            validate_item_review_diagnosis(
                result,
                current_item=state.get("current_item"),
            )
            findings = deepcopy(result["findings"])
            review = {
                "findings": findings,
                "repair_tasks": [],
                "summary": result["summary"],
            }
            validate_item_review(
                review,
                current_item=state.get("current_item"),
            )
            return {
                "state_update": {
                    "current_item_review": review,
                    "review_process_status": "valid",
                    "item_content_status": (
                        "needs_repair"
                        if any(
                            finding.get("severity") == "blocking"
                            for finding in findings
                            if isinstance(finding, Mapping)
                        )
                        else "pass"
                    ),
                    "current_review_request_id": review_request_id,
                    "current_review_item_id": item_id,
                    "current_review_item_version": item_version,
                    "current_review_retry_count": attempt,
                },
                "summary": review["summary"],
                "repair_attempt_count": attempt,
            }
        except ValueError as exc:
            last_error = exc
            if attempt >= max_attempts - 1:
                break
            emit_progress(
                {
                    "type": "output_repair",
                    "retry_kind": _output_error_kind(exc),
                    "job_label": "统一题目审查",
                    "attempt": attempt + 2,
                    "max_attempts": max_attempts,
                    "reason": str(exc),
                }
            )
            input_data = {
                **context,
                "validation_feedback": str(exc),
                "previous_invalid_candidate": _invalid_candidate(result, exc),
            }
    raise ItemReviewProcessError(
        "review_item failed to produce a valid structured review after "
        f"{max_attempts} attempts: {last_error}",
        item_id=item_id,
        item_version=item_version,
        review_request_id=review_request_id,
        attempt_count=max_attempts,
    )


async def execute_virtual_simulation(state: PSJTState) -> dict[str, Any]:
    """Freeze the live candidate pool, then simulate against that exact version."""

    freeze_update = build_item_bank_freeze_update(state)
    simulation_state = {**state, **freeze_update}
    result = await run_virtual_response_simulation(simulation_state)
    simulation_update = result.get("state_update")
    if not isinstance(simulation_update, Mapping):
        raise ValueError("虚拟作答没有返回有效的 state_update")
    return {
        **result,
        "state_update": {
            **freeze_update,
            **dict(simulation_update),
        },
        "summary": (
            f"候选题库已冻结为版本 {freeze_update['item_bank_version']}；"
            + str(result.get("summary") or "虚拟作答完成")
        ),
    }


async def _legacy_execute_item_selection_with_diagnosis(
    state: PSJTState,
) -> dict[str, Any]:
    """LLM diagnoses every item end-to-end; deterministic thresholds are fallback.
    Items are never discarded — every item is either retained or sent to repair.
    """

    frozen = state.get("frozen_item_bank")
    if not isinstance(frozen, list) or not frozen:
        raise ValueError("题目筛选前缺少冻结题库")
    statistics = state.get("item_statistics") or {}
    rounds = dict(state.get("psychometric_repair_rounds") or {})
    max_rounds = int(state.get("max_psychometric_repair_rounds") or 0)

    item_by_id: dict[str, dict[str, Any]] = {}
    for raw_item in frozen:
        if not isinstance(raw_item, Mapping):
            raise ValueError("冻结题库包含无效题目")
        item = dict(raw_item)
        item_id = item.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("冻结题库包含缺少 item_id 的题目")
        if item_id in item_by_id:
            raise ValueError(f"冻结题库包含重复题目：{item_id}")
        item_by_id[item_id] = item

    retained_items: list[dict[str, Any]] = []
    revise_entries: list[dict[str, Any]] = []
    regenerate_entries: list[dict[str, Any]] = []
    reasons: dict[str, str] = {}
    diagnoses: dict[str, dict[str, Any]] = {}
    fallback_ids: list[str] = []

    for item_id, item in item_by_id.items():
        stat = statistics.get(item_id) or {}
        completed_rounds = int(rounds.get(item_id, 0))

        if completed_rounds >= max_rounds:
            retained_items.append(deepcopy(item))
            reasons[item_id] = (
                f"已达到心理测量返修上限 {max_rounds} 轮，保留当前版本"
            )
            continue

        entry = {
            "item_id": item_id,
            "blueprint_cell_id": item.get("blueprint_cell_id"),
            "revision_round": completed_rounds + 1,
        }
        evidence = build_psychometric_repair_evidence(state, entry)
        try:
            diagnosis = await _ainvoke_model(
                psychometric_repair_diagnosis_agent,
                {"input_data": evidence},
                job_label=f"psychometric_repair_diagnosis / {item_id}",
            )
            validate_psychometric_repair_diagnosis(diagnosis, evidence)
            decision = str(diagnosis["decision"])
            diagnoses[item_id] = deepcopy(dict(diagnosis))

            if decision == "retain":
                retained_items.append(deepcopy(item))
                reasons[item_id] = (
                    "LLM 综合诊断：保留 — "
                    + str(diagnosis.get("summary") or "")
                )
            else:
                review = psychometric_diagnosis_to_review(diagnosis, item)
                action = (
                    "revise_item"
                    if decision in {"revise_item", "revise_options"}
                    else "regenerate_item"
                )
                reasons[item_id] = (
                    f"LLM 诊断返回 {decision} — "
                    + str(diagnosis.get("summary") or "")
                )
                resolved_entry = {
                    "item_id": item_id,
                    "blueprint_cell_id": item.get("blueprint_cell_id"),
                    "target_dimension_id": item.get("target_dimension_id"),
                    "action": action,
                    "revision_round": completed_rounds + 1,
                    "review": review,
                    "psychometric_diagnosis": deepcopy(dict(diagnosis)),
                    "diagnosis_status": "completed",
                }
                if action == "revise_item":
                    revise_entries.append(resolved_entry)
                else:
                    regenerate_entries.append(resolved_entry)
        except Exception:
            fallback_ids.append(item_id)
            quality = stat.get("quality_evaluation") or {}
            recommendation = quality.get("recommendation")
            if recommendation == "retain":
                retained_items.append(deepcopy(item))
                reasons[item_id] = (
                    "LLM 诊断失败（确定性回退）：统计达标，保留"
                )
            else:
                if quality.get("discrimination_rating") == "poor":
                    action = "regenerate_item"
                else:
                    action = "revise_item"
                review = {
                    "findings": [
                        {
                            "criterion": "construct_purity",
                            "severity": "blocking",
                            "locus": "response_options",
                            "affected_option_ids": [],
                            "evidence": (
                                "分面内CITC="
                                f"{(stat.get('facet_corrected_item_total_correlation') or {}).get('r', '—')}，"
                                f"难度={stat.get('difficulty', '—')}"
                            ),
                            "problem": "LLM 诊断失败，使用确定性统计阈值回退",
                            "repair_instruction": (
                                "基于统计数据定向修改题目文本，"
                                "优先优化区分度与选项分布"
                            ),
                        }
                    ],
                    "repair_tasks": [],
                    "summary": "确定性回退诊断",
                }
                reasons[item_id] = (
                    "LLM 诊断失败（确定性回退）：统计未达标"
                )
                fallback_entry = {
                    "item_id": item_id,
                    "blueprint_cell_id": item.get("blueprint_cell_id"),
                    "target_dimension_id": item.get("target_dimension_id"),
                    "action": action,
                    "revision_round": completed_rounds + 1,
                    "review": review,
                    "diagnosis_status": "fallback",
                }
                if action == "revise_item":
                    revise_entries.append(fallback_entry)
                else:
                    regenerate_entries.append(fallback_entry)

    selected_items = retained_items if not (revise_entries or regenerate_entries) else []

    selection_status = (
        "ready_for_assembly"
        if not revise_entries and not regenerate_entries
        else "repair_required"
    )
    return {
        "state_update": {
            "selected_items": selected_items,
            "reserve_items": [],
            "items_to_revise": revise_entries,
            "items_to_regenerate": regenerate_entries,
            "items_deferred_for_revision": [],
            "selection_results": {
                "status": selection_status,
                "retained_count": len(retained_items),
                "repair_count": len(revise_entries) + len(regenerate_entries),
                "selected_count": len(selected_items),
                "reserve_count": 0,
                "psychometric_repair_diagnoses": diagnoses,
                "diagnosis_fallback_item_ids": fallback_ids,
                "model_manifest": deepcopy(
                    PSYCHOMETRIC_REASONING_ROLE_MANIFEST
                ),
                "next_effect": {
                    "repair_items": bool(revise_entries or regenerate_entries),
                    "reanalyze_after_bank_change": bool(
                        revise_entries or regenerate_entries
                    ),
                },
            },
            "selection_reasons": reasons,
            "item_pool": [
                deepcopy(dict(item))
                for item in item_by_id.values()
            ],
        },
        "summary": (
            f"LLM 端到端诊断完成：保留 {len(retained_items)} 题，"
            f"返修 {len(revise_entries)} 题，"
            f"重生成 {len(regenerate_entries)} 题"
            + (
                f"（{len(fallback_ids)} 道回退至确定性分类）"
                if fallback_ids
                else ""
            )
        ),
    }


async def execute_item_selection_with_diagnosis(
    state: PSJTState,
) -> dict[str, Any]:
    """Diagnose flagged items and queue all confirmed edits for one item."""

    frozen = state.get("frozen_item_bank")
    if not isinstance(frozen, list) or not frozen:
        raise ValueError("心理测量诊断前缺少冻结题库")
    statistics = state.get("item_statistics") or {}
    rounds = dict(state.get("psychometric_repair_rounds") or {})
    max_rounds = int(state.get("max_psychometric_repair_rounds") or 3)
    prior_fingerprints = {
        str(event.get("diagnosis_fingerprint"))
        for event in state.get("psychometric_repair_history") or []
        if isinstance(event, Mapping) and event.get("diagnosis_fingerprint")
    }
    item_by_id: dict[str, dict[str, Any]] = {}
    for raw_item in frozen:
        if not isinstance(raw_item, Mapping):
            raise ValueError("冻结题库包含无效题目")
        item = deepcopy(dict(raw_item))
        item_id = str(item.get("item_id") or "")
        if not item_id or item_id in item_by_id:
            raise ValueError("冻结题库 item_id 缺失或重复")
        item_by_id[item_id] = item

    retained: list[dict[str, Any]] = []
    # Outer queue: keep every statistically abnormal item here.  Only the
    # first entry is diagnosed in a pass; the rest wait for their turn.
    repair_queue: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    dispositions: dict[str, dict[str, Any]] = deepcopy(
        state.get("item_final_dispositions") or {}
    )
    reasons: dict[str, str] = deepcopy(state.get("selection_reasons") or {})
    diagnoses: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, str] = {}
    diagnosis_call_count = 0
    diagnosis_events: list[dict[str, Any]] = []
    existing_confirmation = state.get("psychometric_repair_confirmation")
    if (
        isinstance(existing_confirmation, Mapping)
        and existing_confirmation.get("status") == "pending"
    ):
        pending_item_id = str(existing_confirmation.get("item_id") or "")
        pending_entry = next(
            (
                deepcopy(entry)
                for entry in state.get("items_to_revise") or []
                if isinstance(entry, Mapping)
                and str(entry.get("item_id")) == pending_item_id
            ),
            None,
        )
        if pending_entry is not None:
            return {
                "state_update": {
                    "selection_results": None,
                    "psychometric_repair_confirmation": deepcopy(
                        dict(existing_confirmation)
                    ),
                    "items_to_revise": [
                        deepcopy(entry)
                        for entry in state.get("items_to_revise") or []
                        if isinstance(entry, Mapping)
                    ],
                    "items_to_regenerate": [
                        deepcopy(entry)
                        for entry in state.get("items_to_regenerate") or []
                        if isinstance(entry, Mapping)
                    ],
                    "selected_items": [],
                    "reserve_items": [],
                },
                "summary": f"等待用户确认单题返修：{pending_item_id}",
            }

    def accept(
        item_id: str,
        item: Mapping[str, Any],
        *,
        warning: str | None,
        reason: str,
    ) -> None:
        retained.append(deepcopy(dict(item)))
        dispositions[item_id] = {
            "status": "accepted" if warning is None else "accepted_with_warning",
            "warning_reason": warning,
            "item_version": item.get("version"),
        }
        reasons[item_id] = reason

    active_item_diagnosed = False
    for item_id, item in item_by_id.items():
        existing_disposition = dispositions.get(item_id)
        # Re-evaluate previously accepted items after any item change because
        # facet-level CITC and distribution statistics can change globally.
        # Explicit warning dispositions (user skipped, deferred, or exhausted
        # budget) remain terminal for this run and are not re-queued.
        if (
            isinstance(existing_disposition, Mapping)
            and existing_disposition.get("warning_reason")
        ):
            retained.append(deepcopy(item))
            continue
        item_statistics = statistics.get(item_id) or {}
        completed_rounds = int(rounds.get(item_id, 0))
        if not item_requires_psychometric_diagnosis(item_statistics):
            accept(item_id, item, warning=None, reason="三类题项指标未触发异常筛查。")
            continue
        if completed_rounds >= max_rounds:
            accept(
                item_id,
                item,
                warning="repair_budget_exhausted",
                reason=f"已完成 {max_rounds} 次原子修改，保留最新版本。",
            )
            continue
        queue_entry = _psychometric_repair_entry(
            item=item,
            statistics=item_statistics,
            revision_round=completed_rounds + 1,
        )
        queue_entry["queue_status"] = "pending_diagnosis"
        repair_queue.append(queue_entry)
        if active_item_diagnosed:
            # The item is already represented in the outer queue.  Do not
            # spend another diagnosis call in this pass.
            continue
        evidence = build_construct_diagnosis_evidence(
            state,
            item_id,
            revision_round=completed_rounds + 1,
        )
        fingerprint = diagnosis_fingerprint(evidence)
        evidence["diagnosis_fingerprint"] = fingerprint
        fingerprints[item_id] = fingerprint
        if fingerprint in prior_fingerprints:
            accept(
                item_id,
                item,
                warning="insufficient_localizable_evidence",
                reason="相同题目版本和统计指纹已经诊断，不重复提交。",
            )
            repair_queue.pop()
            active_item_diagnosed = True
            continue
        try:
            diagnosis_call_count += 1
            diagnosis = await _ainvoke_model(
                psychometric_repair_diagnosis_agent,
                {"input_data": evidence},
                job_label=f"psychometric_repair_diagnosis / {item_id}",
            )
            validate_atomic_repair_advice(diagnosis, evidence)
            diagnoses[item_id] = deepcopy(dict(diagnosis))
            diagnosis_events.append(
                {
                    "event": "psychometric_item_diagnosed",
                    "item_id": item_id,
                    "item_version": item.get("version"),
                    "revision_round": completed_rounds + 1,
                    "diagnosis_fingerprint": fingerprint,
                    "decision": diagnosis.get("decision"),
                    "summary": diagnosis.get("summary"),
                    "repair_task_count": len(repair_tasks_from_advice(diagnosis)),
                }
            )
            if diagnosis["decision"] == "defer":
                components = {
                    component
                    for candidate in diagnosis.get("candidate_diagnoses") or []
                    if isinstance(candidate, Mapping)
                    for component in candidate.get("suspect_components") or []
                }
                if components & {
                    "skeleton", "activation_mechanism", "behavior_evidence", "construct"
                }:
                    warning = "upstream_issue_not_auto_rebuilt"
                elif components & {"simulation", "simulation_or_insufficient_evidence"}:
                    warning = "simulation_inconsistency"
                else:
                    warning = "insufficient_localizable_evidence"
                accept(
                    item_id,
                    item,
                    warning=warning,
                    reason=str(diagnosis.get("summary") or "证据不足，暂不修改。"),
                )
                repair_queue.pop()
                active_item_diagnosed = True
                continue
            diagnosed_entry = {
                "item_id": item_id,
                "blueprint_cell_id": item.get("blueprint_cell_id"),
                "target_dimension_id": item.get("target_dimension_id"),
                "action": "revise_item",
                "revision_round": completed_rounds + 1,
                "atomic_repair_advice": deepcopy(dict(diagnosis)),
                "diagnosis_evidence": deepcopy(evidence),
                "diagnosis_fingerprint": fingerprint,
                "diagnosis_status": "completed",
                "queue_status": "diagnosed",
            }
            repair_queue[-1] = diagnosed_entry
            repairs.append(diagnosed_entry)
            active_item_diagnosed = True
            reasons[item_id] = str(diagnosis.get("summary") or "进入原子返修。")
        except Exception as exc:
            accept(
                item_id,
                item,
                warning="insufficient_localizable_evidence",
                reason=f"诊断没有形成可执行的单点证据：{exc}",
            )
            repair_queue.pop()
            active_item_diagnosed = True
            continue

    status = (
        "repair_confirmation_required"
        if repairs
        else "diagnosis_pending"
        if repair_queue
        else "ready_for_assembly"
    )
    coverage_cells: list[dict[str, Any]] = []
    coverage_passed = True
    item_counts: dict[str, int] = {}
    for item in retained:
        cell_id = item.get("blueprint_cell_id")
        if isinstance(cell_id, str):
            item_counts[cell_id] = item_counts.get(cell_id, 0) + 1
    for cell in (state.get("blueprint") or {}).get("cells") or []:
        if not isinstance(cell, Mapping):
            continue
        cell_id = str(cell.get("cell_id") or "")
        planned = int(cell.get("planned_retention_count") or 0)
        actual = item_counts.get(cell_id, 0)
        passed = actual == planned
        coverage_passed = coverage_passed and passed
        coverage_cells.append(
            {
                "blueprint_cell_id": cell_id,
                "planned_retention_count": planned,
                "selected_count": actual,
                "missing_count": max(0, planned - actual),
                "passed": passed,
            }
        )
    blueprint_coverage = {
        "passed": coverage_passed,
        "cells": coverage_cells,
        "expected_total": sum(
            int(cell.get("planned_retention_count") or 0)
            for cell in coverage_cells
        ),
        "selected_total": len(retained),
    }
    return {
        "state_update": {
            "selected_items": [] if repair_queue else retained,
            "reserve_items": [],
            "items_to_revise": repair_queue,
            "items_to_regenerate": [],
            "items_deferred_for_revision": [],
            "selection_results": None if repair_queue else {
                "status": status,
                "retained_count": len(retained),
                "repair_count": len(repair_queue),
                "selected_count": len(retained),
                "reserve_count": 0,
                "psychometric_repair_diagnoses": diagnoses,
                "diagnosis_evidence_fingerprints": fingerprints,
                "diagnosis_call_count": diagnosis_call_count,
                "final_dispositions": dispositions,
                "model_manifest": deepcopy(PSYCHOMETRIC_REASONING_ROLE_MANIFEST),
                "next_effect": {
                    "repair_items": bool(repair_queue),
                    "reanalyze_after_bank_change": bool(repair_queue),
                },
            },
            "blueprint_coverage": blueprint_coverage,
            "selection_reasons": reasons,
            "item_final_dispositions": dispositions,
            "psychometric_repair_confirmation": (
                {
                    "status": "pending",
                    "item_id": repair_queue[0]["item_id"],
                    "revision_round": repair_queue[0]["revision_round"],
                    "diagnosis_fingerprint": repair_queue[0].get("diagnosis_fingerprint"),
                    "atomic_repair_advice": deepcopy(
                        repair_queue[0].get("atomic_repair_advice")
                    ),
                    "diagnosis_evidence": deepcopy(
                        repair_queue[0].get("diagnosis_evidence")
                    ),
                }
                if repairs and isinstance(repair_queue[0].get("atomic_repair_advice"), Mapping)
                else None
            ),
            "psychometric_repair_history": [
                *deepcopy(state.get("psychometric_repair_history") or []),
                *diagnosis_events,
            ],
            "item_pool": [deepcopy(item) for item in item_by_id.values()],
        },
        "summary": (
            f"构念约束诊断完成：当前保留 {len(retained)} 题，"
            f"外层待处理队列 {len(repair_queue)} 题。"
        ),
    }


def _construct_domain_summary(
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Project stable domain identity without repeating all facet content."""

    fields = (
        "inventory_id",
        "inventory_name",
        "inventory_version",
        "review_status",
        "selection_level",
        "domain_id",
        "domain_name",
        "domain_name_en",
        "profile_hash",
    )
    return {
        field: deepcopy(profile.get(field))
        for field in fields
        if profile.get(field) is not None
    }


def _other_facet_boundaries(
    profile: Mapping[str, Any],
    current_facet_id: str,
) -> list[dict[str, Any]]:
    """Keep only the semantic exclusion boundary of non-target facets."""

    fields = (
        "facet_id",
        "facet_name",
        "definition",
        "high_behavior",
        "low_behavior",
    )
    return [
        {
            field: deepcopy(facet.get(field))
            for field in fields
            if facet.get(field) is not None
        }
        for facet in profile.get("facets") or []
        if isinstance(facet, Mapping)
        and facet.get("facet_id") != current_facet_id
    ]


def _skeleton_review_construct_context(
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Provide one non-duplicated global construct view to the reviewer."""

    return {
        "domain_summary": _construct_domain_summary(profile),
        "facets": [
            deepcopy(dict(facet))
            for facet in profile.get("facets") or []
            if isinstance(facet, Mapping)
        ],
    }


async def _fill_compact_slots(
    state: PSJTState,
    blueprint: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    valid_seed: Mapping[str, Mapping[str, Any]] | None = None,
    target_ids: set[str] | None = None,
    attempts_used: int = 0,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Fill fixed slots one at a time without exposing their IDs to the model.

    Slot identity is workflow state, not generated content.  Each model call
    therefore returns one anonymous skeleton payload; this function attaches
    it to the active program-owned specification ID and performs all duplicate
    and schema validation locally.
    """

    all_cell_ids = {
        slot["specification_id"]
        for slot in blueprint.get("slots") or []
        if isinstance(slot, Mapping)
        and slot.get("blueprint_cell_id") == cell.get("cell_id")
    }
    targets = set(target_ids or all_cell_ids) & all_cell_ids
    valid = {
        key: deepcopy(dict(value))
        for key, value in (valid_seed or {}).items()
        if isinstance(value, Mapping)
    }
    profile = blueprint["construct_profile_snapshot"]
    design = resolve_blueprint_design(blueprint, cell)
    facet = deepcopy(design["facet"])
    facet.pop("behavior_evidence", None)
    behavior = deepcopy(design["behavior_evidence"])
    behavior.pop("source_item_ids", None)
    slot_order = [
        str(slot["specification_id"])
        for slot in blueprint.get("slots") or []
        if isinstance(slot, Mapping)
        and slot.get("specification_id") in targets
    ]
    max_attempt_depth = max(1, attempts_used)
    for ordinal, specification_id in enumerate(slot_order, start=1):
        if specification_id in valid:
            continue
        last_error: ValueError | None = None
        last_problems: list[str] = []
        for attempt in range(1, MAX_ITEM_SKELETON_ATTEMPTS + 1):
            max_attempt_depth = max(max_attempt_depth, attempt)
            input_data: dict[str, Any] = {
                "test_specification": state.get("test_specification"),
                "domain_summary": _construct_domain_summary(profile),
                "current_facet": facet,
                "behavior_evidence": behavior,
                "activation_mechanism": design["activation_mechanism"],
                "situation": design["situation"],
            }
            if last_error is not None or last_problems:
                input_data["validation_feedback"] = {
                    "problems": last_problems or [str(last_error)],
                    "attempt": attempt,
                    "max_attempts": MAX_ITEM_SKELETON_ATTEMPTS,
                    "instruction": (
                        "只修正当前匿名骨架；不要返回任何题号或映射键"
                    ),
                }
            try:
                result = await _ainvoke_model(
                    compact_skeleton_agent,
                    {"input_data": input_data},
                    job_label=(
                        f"心理骨架-{facet['facet_name']}-{ordinal}"
                    ),
                )
                update = result.get("state_update")
                if not isinstance(update, Mapping):
                    raise ValueError("骨架输出缺少 state_update")
                candidate = update.get("item_skeleton")
                if not isinstance(candidate, Mapping):
                    raise ValueError("骨架输出必须包含一个 item_skeleton 对象")
            except ValueError as exc:
                last_error = exc
                last_problems = []
            else:
                check = classify_compact_skeletons(
                    blueprint,
                    {specification_id: candidate},
                )
                if specification_id in check["valid"]:
                    valid[specification_id] = deepcopy(
                        check["valid"][specification_id]
                    )
                    last_error = None
                    last_problems = []
                    break
                last_error = None
                last_problems = list(
                    check["invalid"].get(
                        specification_id,
                        ["当前骨架未通过程序校验"],
                    )
                )
            if attempt < MAX_ITEM_SKELETON_ATTEMPTS:
                emit_progress(
                    {
                        "type": "output_repair",
                        "retry_kind": (
                            _output_error_kind(last_error)
                            if last_error is not None
                            else "business_validation"
                        ),
                        "job_label": f"心理骨架-{facet['facet_name']}",
                        "attempt": attempt + 1,
                        "max_attempts": MAX_ITEM_SKELETON_ATTEMPTS,
                        "reason": (
                            str(last_error)
                            if last_error is not None
                            else "；".join(last_problems)
                        ),
                    }
                )
        else:
            details = last_problems or [str(last_error or "未知错误")]
            raise ValueError(
                "当前固定槽位经过 "
                f"{MAX_ITEM_SKELETON_ATTEMPTS} 次尝试仍未形成有效骨架："
                + "；".join(details)
            )
    return valid, max_attempt_depth


async def execute_fixed_blueprint(state: PSJTState) -> dict:
    """Prepare evidence, expansion, and the immutable two-way table."""

    specification = state.get("test_specification")
    if not isinstance(specification, Mapping):
        raise ValueError("建立题目计划前缺少 TestSpecification")
    profile = resolve_specification_profile(specification)
    corpus = load_ipip_corpus()
    bundles = {}
    for facet in profile["facets"]:
        facet_id = str(facet["facet_id"])
        bundles[facet_id] = await ensure_behavior_evidence(facet_id, corpus)
    profile = attach_behavior_evidence(profile, bundles)
    retention_total = int(specification["final_item_count"])
    situation_quotas = distribute_situation_quotas(
        retention_total, len(profile["facets"])
    )
    expansions = []
    for facet, required_situation_count in zip(
        profile["facets"], situation_quotas, strict=True
    ):
        expansions.append(
            await ensure_facet_expansion(
                run_id=state["run_id"],
                facet=facet,
                behavior_evidence=facet["behavior_evidence"],
                target_population=str(specification["target_population"]),
                output_language=str(specification["output_language"]),
                required_situation_count=required_situation_count,
            )
        )
    generation_total = required_generation_total(retention_total)
    proposal = await propose_blueprint_rows(
        profile=profile,
        expansions=expansions,
        generation_total=generation_total,
        retention_total=retention_total,
    )
    blueprint = build_generation_blueprint(
        specification,
        profile,
        state["run_id"],
        expansions=expansions,
        proposal=proposal,
    )
    errors = validate_generation_blueprint(blueprint, specification)
    if errors:
        raise ValueError(format_blueprint_errors_for_user(errors))
    return {
        "state_update": {
            "construct_profile": profile,
            "blueprint": blueprint,
        },
        "summary": (
            f"题目计划引用 {profile['inventory_name']} "
            f"{profile['domain_name']}，包含 {len(profile['facets'])} 个 "
            f"facet、{planned_generation_count(blueprint)} 个固定槽位，"
            f"计划最终保留 {specification['final_item_count']} 题。"
        ),
        "repair_attempt_count": 0,
    }


async def _prepare_current_slot_skeleton(
    state: PSJTState,
) -> dict[str, Any]:
    """Generate one skeleton and apply program-owned deterministic checks."""

    blueprint = state.get("blueprint")
    item_specification = state.get("current_item_specification")
    if not isinstance(blueprint, Mapping) or not isinstance(
        item_specification, Mapping
    ):
        raise ValueError("当前题目槽缺少固定蓝图或槽位信息")
    specification_id = str(item_specification["specification_id"])
    existing = {
        str(key): deepcopy(dict(value))
        for key, value in (state.get("item_skeletons") or {}).items()
        if isinstance(value, Mapping)
    }
    skeleton = existing.get(specification_id)
    history = [
        deepcopy(dict(entry))
        for entry in (state.get("skeleton_review_history") or {}).get(
            specification_id, []
        )
        if isinstance(entry, Mapping)
    ]

    async def generate() -> dict[str, Any]:
        seed = {
            key: value for key, value in existing.items()
            if key != specification_id
        }
        cell = next(
            candidate
            for candidate in blueprint["cells"]
            if candidate["cell_id"]
            == item_specification["blueprint_cell_id"]
        )
        generated, _ = await _fill_compact_slots(
            state,
            blueprint,
            cell,
            valid_seed=seed,
            target_ids={specification_id},
        )
        return generated[specification_id]

    if skeleton is None:
        try:
            skeleton = await generate()
        except ValueError as exc:
            history.append(
                {
                    "round": len(history) + 1,
                    "mode": "program_validation_failed",
                    "skeleton": None,
                    "validation_error": str(exc),
                }
            )
            emit_progress(
                {
                    "type": "skeleton_slot_failed",
                    "specification_id": specification_id,
                    "history": deepcopy(history),
                    "final_reason": str(exc),
                }
            )
            raise SkeletonDevelopmentFailure(
                specification_id,
                history,
                str(exc),
            ) from exc

    history.append(
        {
            "round": len(history) + 1,
            "mode": "program_validation_passed",
            "skeleton": deepcopy(dict(skeleton)),
            "validation_summary": (
                "Schema、固定槽位映射及基础重复规则通过；"
                "未执行独立 LLM 骨架审核"
            ),
        }
    )

    committed_skeletons = {**existing, specification_id: skeleton}
    rows = materialize_item_specifications(
        blueprint,
        {specification_id: skeleton},
    )
    if len(rows) != 1:
        raise ValueError("当前心理骨架无法映射为唯一题目规格")
    current_row = rows[0]
    all_rows = [
        deepcopy(dict(row))
        for row in state.get("item_specifications") or []
        if isinstance(row, Mapping)
        and row.get("specification_id") != specification_id
    ]
    all_rows.append(current_row)
    return {
        "item_skeletons": committed_skeletons,
        "skeleton_reviews": {
            key: value
            for key, value in (state.get("skeleton_reviews") or {}).items()
            if key != specification_id
        },
        "skeleton_review_history": {
            **(state.get("skeleton_review_history") or {}),
            specification_id: history,
        },
        "item_specifications": all_rows,
        "current_item_specification": current_row,
    }


def _build_option_evidence_for_repair(
    state: Mapping[str, Any],
    active_psychometric_repair: Mapping[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Extract per-option psychometric evidence for the repair LLM."""
    if not isinstance(active_psychometric_repair, Mapping):
        return None
    statistics = (state.get("item_statistics") or {}).get(
        active_psychometric_repair.get("item_id")
    ) or {}
    option_stats = statistics.get("option_statistics") or {}
    if not option_stats:
        return None
    result: list[dict[str, Any]] = []
    for option_id in sorted(option_stats):
        opt_stat = option_stats.get(option_id) or {}
        result.append({
            "option_id": option_id,
            "selection_count": opt_stat.get("count"),
            "selection_rate": opt_stat.get("rate"),
            "score": opt_stat.get("score"),
        })
    return result


async def _execute_psychometric_repair_batch(
    *,
    action: str,
    route: PSJTRouteDecision,
    state: PSJTState,
    item_specification: Mapping[str, Any] | None,
    active_psychometric_repair: Mapping[str, Any],
    atomic_advice: Mapping[str, Any],
    diagnosis_evidence: Mapping[str, Any],
    program_update: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Apply every confirmed task for one item before any re-simulation.

    Each task still gets its own narrowly scoped repair-model call.  The
    intermediate item is kept local until all tasks succeed, so a failed task
    cannot leave a partially repaired item in the workflow.  The caller then
    commits one new item version and clears statistics once, which triggers one
    subsequent virtual re-test for the completed batch.
    """

    tasks = repair_tasks_from_advice(atomic_advice)
    if not tasks:
        return None
    working_item = deepcopy(state.get("current_item") or {})
    if not working_item:
        raise ValueError("psychometric repair batch requires current_item")
    base_version = int(working_item.get("version") or 0)
    normal_constraints = diagnosis_evidence.get("normal_constraints")
    option_evidence = diagnosis_evidence.get("option_evidence")
    total_attempts = 0

    for task_index, task in enumerate(tasks, start=1):
        diagnosis_id = str(task.get("diagnosis_id") or f"D{task_index}")
        task_advice = deepcopy(dict(atomic_advice))
        task_advice["selected_diagnosis_id"] = diagnosis_id
        task_advice["atomic_edit"] = deepcopy(task.get("atomic_edit"))
        if "repair_tasks" in atomic_advice:
            task_advice["repair_tasks"] = [deepcopy(dict(task))]
        input_data: dict[str, Any] = {
            "action": action,
            "state": build_item_model_state({**state, "current_item": working_item}),
            "generation_context": build_item_generation_context(
                {**state, "current_item": working_item}
            ),
            "blocking_findings": [],
            "repair_source": "psychometric_diagnosis",
            "atomic_repair_advice": task_advice,
            "normal_constraints": normal_constraints,
            "option_evidence": option_evidence,
            "required_context_category": (
                item_specification.get("context_category")
                if isinstance(item_specification, Mapping)
                else None
            ),
            "validation_feedback": None,
            "previous_invalid_candidate": None,
        }
        task_error: ValueError | None = None
        task_succeeded = False
        for repair_attempt in range(MAX_ITEM_OUTPUT_CANDIDATES):
            total_attempts += 1
            result: Any = None
            try:
                result = await _ainvoke_model(
                    psychometric_item_repair_agent,
                    {"input_data": input_data},
                    job_label=(
                        f"{action} / {route.get('target_item_id') or 'item'}"
                        f" / {diagnosis_id}"
                    ),
                )
                if not isinstance(result, Mapping):
                    raise ValueError("Agent output must be an object")
                proposed_update = result.get("state_update")
                if not isinstance(proposed_update, dict):
                    raise ValueError("Agent output missing valid state_update")
                validate_atomic_item_patch(
                    proposed_update,
                    working_item,
                    task_advice,
                )
                proposed_update = canonicalize_item_agent_update(
                    action,
                    proposed_update,
                    specification=state.get("test_specification"),
                    blueprint_cell=state.get("current_blueprint_cell"),
                    item_specification=item_specification,
                    previous_item=working_item,
                )
                validate_item_agent_update(
                    action,
                    proposed_update,
                    target_item_id=route.get("target_item_id"),
                    target_blueprint_cell_id=route.get("target_blueprint_cell_id"),
                    specification=state.get("test_specification"),
                    blueprint_cell=state.get("current_blueprint_cell"),
                    item_specification=item_specification,
                    previous_item=working_item,
                )
                working_item = deepcopy(proposed_update["current_item"])
                task_succeeded = True
                break
            except ValueError as exc:
                task_error = exc
                if repair_attempt >= MAX_ITEM_OUTPUT_CANDIDATES - 1:
                    break
                emit_progress(
                    {
                        "type": "output_repair",
                        "retry_kind": _output_error_kind(exc),
                        "job_label": f"{action} / {diagnosis_id}",
                        "attempt": repair_attempt + 2,
                        "max_attempts": MAX_ITEM_OUTPUT_CANDIDATES,
                        "reason": str(exc),
                    }
                )
                input_data = {
                    **input_data,
                    "validation_feedback": str(exc),
                    "previous_invalid_candidate": _invalid_candidate(
                        result,
                        exc,
                        state_update_only=True,
                    ),
                }
        if not task_succeeded:
            raise ValueError(
                f"{action} task {diagnosis_id} did not produce a valid patch: "
                f"{task_error}"
            )

    # The individual task results are deliberately kept in memory.  They are
    # one psychometric repair transaction, so the persisted item receives one
    # version bump, not one bump per option/task.  This also lets the outer
    # execute-node validator compare the committed candidate with the original
    # item version without rejecting a valid multi-task repair.
    working_item["version"] = base_version + 1

    return {
        "state_update": {
            "current_item": working_item,
            **dict(program_update),
        },
        "repair_attempt_count": total_attempts,
        "summary": f"同一题已完成 {len(tasks)} 条定向修改，随后统一重新施测",
    }


async def execute_item_action_with_repair(
    action: str,
    route: PSJTRouteDecision,
    state: PSJTState,
) -> dict[str, Any]:
    """对题目 Agent 的无效结构输出进行最多3次有界修复。"""

    active_psychometric_repair = state.get("active_psychometric_repair")
    agent = (
        psychometric_item_repair_agent
        if action in {"revise_item", "regenerate_item"}
        and isinstance(active_psychometric_repair, Mapping)
        else AGENT_MAP[action]
    )
    program_update: dict[str, Any] = {}
    item_specification = state.get("current_item_specification")
    if (
        (
            action == "generate_item"
            or (
                action == "regenerate_item"
                and bool(state.get("current_skeleton_repair_required"))
            )
        )
        and isinstance(state.get("blueprint"), Mapping)
        and state["blueprint"].get("version") == 7
        and (
        not isinstance(item_specification, Mapping)
        or "activation_mechanism" not in item_specification
        )
    ):
        program_update = await _prepare_current_slot_skeleton(state)
        state = {**state, **program_update}
        item_specification = state["current_item_specification"]
    current_review = state.get("current_item_review")
    blocking_findings = [
        deepcopy(finding)
        for finding in (current_review or {}).get("findings") or []
        if isinstance(finding, Mapping)
        and finding.get("severity") == "blocking"
    ]
    # Resume older checkpoints safely after behavioral levels became immutable.
    # Legacy level-edit requests are realized as option-text repairs.
    for finding in blocking_findings:
        if finding.get("locus") == "behavioral_level":
            finding["locus"] = "response_options"
            finding["repair_instruction"] = (
                str(finding.get("repair_instruction") or "")
                + " Keep behavioral_level and scoring_key unchanged; rewrite "
                "the named option text to realize its fixed level."
            )
        for edit in finding.get("required_edits") or []:
            if isinstance(edit, dict) and edit.get("field") == "behavioral_level":
                edit["field"] = "response_options"
    atomic_advice = (
        deepcopy(active_psychometric_repair.get("atomic_repair_advice"))
        if isinstance(active_psychometric_repair, Mapping)
        and isinstance(
            active_psychometric_repair.get("atomic_repair_advice"), Mapping
        )
        else None
    )
    diagnosis_evidence = (
        deepcopy(active_psychometric_repair.get("diagnosis_evidence"))
        if isinstance(active_psychometric_repair, Mapping)
        and isinstance(active_psychometric_repair.get("diagnosis_evidence"), Mapping)
        else None
    )
    if (
        action in {"revise_item", "regenerate_item"}
        and not blocking_findings
        and atomic_advice is None
    ):
        raise ValueError(f"{action} 缺少 blocking 审题意见")
    if (
        action in {"revise_item", "regenerate_item"}
        and isinstance(active_psychometric_repair, Mapping)
        and isinstance(atomic_advice, Mapping)
        and repair_tasks_from_advice(atomic_advice)
    ):
        return await _execute_psychometric_repair_batch(
            action=action,
            route=route,
            state=state,
            item_specification=(
                item_specification if isinstance(item_specification, Mapping) else None
            ),
            active_psychometric_repair=active_psychometric_repair,
            atomic_advice=atomic_advice,
            diagnosis_evidence=(diagnosis_evidence or {}),
            program_update=program_update,
        )
    model_state = build_item_model_state(state)
    if blocking_findings:
        model_state = {
            **model_state,
            "current_item_review": None,
        }
    input_data: dict[str, Any] = {
        "action": action,
        "state": model_state,
        "generation_context": build_item_generation_context(state),
        "blocking_findings": blocking_findings,
        "repair_source": (
            "psychometric_diagnosis"
            if atomic_advice is not None
            else "content_review"
        ),
        "atomic_repair_advice": atomic_advice,
        "normal_constraints": (
            diagnosis_evidence.get("normal_constraints")
            if diagnosis_evidence is not None
            else None
        ),
        "option_evidence": (
            diagnosis_evidence.get("option_evidence")
            if diagnosis_evidence is not None
            else _build_option_evidence_for_repair(state, active_psychometric_repair)
        ),
        "required_context_category": (
            item_specification.get("context_category")
            if isinstance(item_specification, dict)
            else None
        ),
        "validation_feedback": None,
        "previous_invalid_candidate": None,
    }
    last_error: ValueError | None = None
    max_candidates = MAX_ITEM_OUTPUT_CANDIDATES
    accumulated_patch: dict[str, Any] | None = None
    for repair_attempt in range(max_candidates):
        result: Any = None
        try:
            result = await _ainvoke_model(
                agent,
                {"input_data": input_data},
                job_label=(
                    f"{action} / {route.get('target_item_id') or '新题'}"
                ),
            )
            if not isinstance(result, dict):
                raise ValueError("Agent 输出必须是对象")
            proposed_update = result.get("state_update")
            if not isinstance(proposed_update, dict):
                raise ValueError("Agent 输出缺少有效的 state_update")
            if action in {"revise_item", "regenerate_item"}:
                scenario_update = proposed_update.get("scenario_update")
                option_updates = proposed_update.get("option_updates")
                if atomic_advice is not None:
                    validate_atomic_item_patch(
                        proposed_update,
                        state.get("current_item") or {},
                        atomic_advice,
                    )
                elif isinstance(accumulated_patch, dict):
                    merged_options = {
                        str(patch.get("option_id")): deepcopy(patch)
                        for patch in accumulated_patch.get("option_updates") or []
                        if isinstance(patch, Mapping) and patch.get("option_id")
                    }
                    if isinstance(option_updates, list):
                        for patch in option_updates:
                            if isinstance(patch, Mapping) and patch.get("option_id"):
                                merged_options[str(patch["option_id"])] = deepcopy(
                                    dict(patch)
                                )
                    proposed_update = {
                        "scenario_update": (
                            scenario_update
                            if scenario_update is not None
                            else accumulated_patch.get("scenario_update")
                        ),
                        "option_updates": list(merged_options.values()),
                    }
            proposed_update = canonicalize_item_agent_update(
                action,
                proposed_update,
                specification=state.get("test_specification"),
                blueprint_cell=state.get("current_blueprint_cell"),
                item_specification=item_specification,
                previous_item=state.get("current_item"),
            )
            if action in {"revise_item", "regenerate_item"}:
                accumulated_patch = {
                    "scenario_update": (
                        proposed_update["current_item"].get("scenario")
                        if proposed_update["current_item"].get("scenario")
                        != (state.get("current_item") or {}).get("scenario")
                        else None
                    ),
                    "option_updates": [
                        {
                            "option_id": option.get("option_id"),
                            "text": option.get("text"),
                        }
                        for option in proposed_update["current_item"].get(
                            "response_options"
                        )
                        or []
                        if isinstance(option, Mapping)
                        and next(
                            (
                                prior.get("text")
                                for prior in (
                                    (state.get("current_item") or {}).get(
                                        "response_options"
                                    )
                                    or []
                                )
                                if isinstance(prior, Mapping)
                                and prior.get("option_id")
                                == option.get("option_id")
                            ),
                            None,
                        )
                        != option.get("text")
                    ],
                }
            result = {
                **result,
                "state_update": proposed_update,
            }
            validate_item_agent_update(
                action,
                proposed_update,
                target_item_id=route.get("target_item_id"),
                target_blueprint_cell_id=route.get(
                    "target_blueprint_cell_id"
                ),
                specification=state.get("test_specification"),
                blueprint_cell=state.get("current_blueprint_cell"),
                item_specification=item_specification,
                previous_item=state.get("current_item"),
            )
            return {
                **result,
                "state_update": {
                    **proposed_update,
                    **program_update,
                },
                "repair_attempt_count": repair_attempt,
            }
        except ValueError as exc:
            last_error = exc
            if repair_attempt >= max_candidates - 1:
                break
            emit_progress(
                {
                    "type": "output_repair",
                    "retry_kind": _output_error_kind(exc),
                    "job_label": action,
                    "attempt": repair_attempt + 2,
                    "max_attempts": max_candidates,
                    "reason": str(exc),
                }
            )
            input_data = {
                **input_data,
                "validation_feedback": str(exc),
                "previous_invalid_candidate": _invalid_candidate(
                    result,
                    exc,
                    state_update_only=True,
                ),
            }
            if accumulated_patch is not None:
                input_data["previous_invalid_candidate"] = deepcopy(
                    accumulated_patch
                )

    raise ValueError(
        f"{action} 经过 {max_candidates} 次候选输出后"
        f"仍未通过结构校验：{last_error}"
    )


async def execute_agent(
    route: PSJTRouteDecision,
    state: PSJTState,
) -> dict:
    action = route["next_action"]
    if action == "build_blueprint":
        return await execute_fixed_blueprint(state)
    if action == "review_item":
        return await execute_item_review(state)
    if action in {"generate_item", "regenerate_item", "revise_item"}:
        return await execute_item_action_with_repair(action, route, state)
    if action == "simulate_responses":
        return await execute_virtual_simulation(state)
    if action == "analyze_psychometrics":
        return await asyncio.to_thread(run_psychometric_analysis, state)
    if action == "select_items":
        return await execute_item_selection_with_diagnosis(state)
    if action == "assemble_test":
        return await asyncio.to_thread(run_test_assembly, state)
    if action == "review_test":
        return await asyncio.to_thread(run_test_review, state)
    if action == "rescore_test":
        return await asyncio.to_thread(run_test_rescore, state)
    if action == "generate_reports":
        return await asyncio.to_thread(run_report_generation, state)
    agent = AGENT_MAP.get(action)
    if agent is None:
        raise ValueError(f"没有为任务 {action!r} 注册对应的 Agent")
    input_data = {
        "state": build_requirement_model_state(state),
        "target_item_id": route.get("target_item_id"),
        "target_blueprint_cell_id": route.get("target_blueprint_cell_id"),
    }
    result = await _ainvoke_model(
        agent,
        {"input_data": input_data},
        job_label=action,
    )
    return result
