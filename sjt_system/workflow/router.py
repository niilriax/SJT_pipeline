"""Top-level workflow routing policy and completion guards."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from time import perf_counter
from typing import Any

from sjt_system.authoring.bank import (
    build_blueprint_retention_gaps,
    item_bank_snapshot_is_current,
)
from sjt_system.authoring.blueprint import select_next_blueprint_cell
from sjt_system.authoring.context import select_item_specification
from sjt_system.authoring.generation_plan import GENERATION_BLUEPRINT_VERSION
from sjt_system.delivery.lifecycle import (
    evaluate_completion,
    psychometric_results_are_current,
)
from sjt_system.runtime.trace import utc_timestamp
from sjt_system.state import PSJTState


def _fixed_blueprint_gap_failure(
    state: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Stop instead of letting any model or recovery path invent new IDs."""

    message = (
        f"{reason}。固定蓝图中的题号已经耗尽；系统不会在运行中创建补题题号。"
        "请在原题号内继续重写，或返回蓝图阶段重新确认完整细目表。"
    )
    return {
        "status": "failed",
        "errors": [
            *(state.get("errors") or []),
            {"action": "fixed_blueprint_gap", "message": message},
        ],
        "execution_history": [
            *(state.get("execution_history") or []),
            {
                "event_id": (
                    f"{state.get('run_id')}:{state.get('step_count')}:"
                    "router:fixed_blueprint_gap"
                ),
                "run_id": state.get("run_id"),
                "step": state.get("step_count"),
                "node": "router",
                "action": "fixed_blueprint_gap",
                "event_type": "failed",
                "reason": message,
                "recorded_at": utc_timestamp(),
                "duration_ms": 0,
                "error": message,
            },
        ],
    }


def _next_psychometric_repair(
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    entries = [
        deepcopy(dict(entry))
        for entry in [
            *(state.get("items_to_regenerate") or []),
            *(state.get("items_to_revise") or []),
        ]
        if isinstance(entry, Mapping) and entry.get("item_id")
    ]
    if not entries:
        return None
    order = {
        str(item.get("item_id")): index
        for index, item in enumerate(state.get("frozen_item_bank") or [])
        if isinstance(item, Mapping) and item.get("item_id")
    }
    entries.sort(
        key=lambda entry: (
            order.get(str(entry["item_id"]), len(order)),
            str(entry["item_id"]),
        )
    )
    return entries[0]


def _psychometric_confirmation_is_pending(state: Mapping[str, Any]) -> bool:
    confirmation = state.get("psychometric_repair_confirmation")
    return isinstance(confirmation, Mapping) and not confirmation.get("decision")


def _psychometric_confirmation_is_approved(state: Mapping[str, Any]) -> bool:
    confirmation = state.get("psychometric_repair_confirmation")
    return (
        isinstance(confirmation, Mapping)
        and confirmation.get("decision") == "approve"
    )


def _has_batchable_psychometric_repairs(state: Mapping[str, Any]) -> bool:
    """Whether the diagnosed queue can be handed to parallel item workers."""

    for entry in [
        *(state.get("items_to_regenerate") or []),
        *(state.get("items_to_revise") or []),
    ]:
        if not isinstance(entry, Mapping):
            continue
        advice = entry.get("atomic_repair_advice")
        if (
            isinstance(advice, Mapping)
            and advice.get("decision") == "repair"
            and advice.get("repair_tasks")
        ):
            return True
    return False


def _stage_psychometric_repair(
    state: Mapping[str, Any],
    repair: Mapping[str, Any],
) -> dict[str, Any]:
    item_id = str(repair["item_id"])
    current_item = next(
        (
            deepcopy(dict(item))
            for item in state.get("item_pool") or []
            if isinstance(item, Mapping) and item.get("item_id") == item_id
        ),
        None,
    )
    if current_item is None:
        raise ValueError(f"心理测量返修找不到题目 {item_id}")
    cell_id = current_item.get("blueprint_cell_id")
    blueprint_cell = next(
        (
            deepcopy(dict(cell))
            for cell in (state.get("blueprint") or {}).get("cells") or []
            if isinstance(cell, Mapping) and cell.get("cell_id") == cell_id
        ),
        None,
    )
    if blueprint_cell is None:
        raise ValueError(f"心理测量返修找不到蓝图单元 {cell_id}")
    item_specification = next(
        (
            deepcopy(dict(specification))
            for specification in state.get("item_specifications") or []
            if isinstance(specification, Mapping)
            and specification.get("specification_id") == item_id
        ),
        None,
    )
    if item_specification is None:
        item_specification = {
            "specification_id": item_id,
            "blueprint_cell_id": cell_id,
            "target_dimension_id": current_item.get("target_dimension_id"),
            "context_category": current_item.get("context_category"),
            "context_seed": current_item.get("scenario"),
            "avoid_scenario_patterns": [],
            "avoid_response_patterns": [],
        }
    advice = repair.get("atomic_repair_advice")
    evidence = repair.get("diagnosis_evidence")
    if not isinstance(advice, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError(f"心理测量返修任务 {item_id} 缺少结构化修改意见")
    return {
        "current_item": current_item,
        "current_blueprint_cell": blueprint_cell,
        "current_item_specification": item_specification,
        "current_item_review": None,
        # This flag describes whether the psychometric candidate has already
        # received its one bounded content-cleanup pass. Staging the candidate
        # is not itself a repair attempt.
        "current_item_repair_attempted": False,
        "current_item_repair_failure": None,
        "current_item_revision_count": 0,
        "current_item_rewrite_count": 0,
        "active_psychometric_repair": {
            **deepcopy(dict(repair)),
            "baseline_item": deepcopy(current_item),
            "baseline_profile": deepcopy(
                (state.get("item_pattern_profiles") or {}).get(item_id)
            ),
            "baseline_analysis_snapshot": (
                {
                    "frozen_item_bank": deepcopy(
                        state.get("frozen_item_bank")
                    ),
                    "item_bank_id": state.get("item_bank_id"),
                    "item_bank_version": state.get("item_bank_version"),
                    "item_bank_fingerprint": state.get(
                        "item_bank_fingerprint"
                    ),
                    "item_bank_frozen_at": state.get("item_bank_frozen_at"),
                    "virtual_response_data_ref": state.get(
                        "virtual_response_data_ref"
                    ),
                    "virtual_response_summary": deepcopy(
                        state.get("virtual_response_summary")
                    ),
                    "virtual_response_item_bank_id": state.get(
                        "virtual_response_item_bank_id"
                    ),
                    "virtual_response_item_bank_version": state.get(
                        "virtual_response_item_bank_version"
                    ),
                    "item_statistics": deepcopy(
                        state.get("item_statistics")
                    ),
                    "test_statistics": deepcopy(state.get("test_statistics")),
                }
                if state.get("virtual_response_data_ref")
                and state.get("item_statistics")
                and state.get("test_statistics")
                else None
            ),
        },
        "psychometric_repair_confirmation": None,
    }


def _blueprint_review_allows_item_generation(state: Mapping[str, Any]) -> bool:
    blueprint = state.get("blueprint")
    return (
        isinstance(blueprint, Mapping)
        and blueprint.get("version") == GENERATION_BLUEPRINT_VERSION
    )


async def router_node(state: PSJTState) -> dict:
    """读取 State 并选择下一项任务。"""
    started_at = perf_counter()
    if state["step_count"] >= state["max_steps"]:
        message = "测试达到最大执行步数，流程已停止"
        return {
            "status": "failed",
            "errors": [
                *state["errors"],
                {
                    "action": None,
                    "message": message,
                },
            ],
            "execution_history": [
                *state["execution_history"],
                {
                    "event_id": f'{state["run_id"]}:{state["step_count"]}:router:failed',
                    "run_id": state["run_id"],
                    "step": state["step_count"],
                    "node": "router",
                    "action": "stop",
                    "event_type": "failed",
                    "reason": message,
                    "recorded_at": utc_timestamp(),
                    "duration_ms": round((perf_counter() - started_at) * 1000),
                    "error": message,
                },
            ],
        }

    psychometrics_complete_before_route = psychometric_results_are_current(
        state
    )
    selection_status = (
        (state.get("selection_results") or {}).get("status")
        if isinstance(state.get("selection_results"), Mapping)
        else None
    )
    test_review_before_route = state.get("test_review_result")
    test_review_decision_before_route = (
        test_review_before_route.get("decision")
        if isinstance(test_review_before_route, Mapping)
        else None
    )
    final_outputs_exist = (
        isinstance(state.get("final_test"), Mapping)
        and bool(state.get("item_database_ref"))
        and isinstance(state.get("technical_report"), Mapping)
        and isinstance(state.get("virtual_respondent_report"), Mapping)
    )
    responses_bound_to_current_bank = (
        item_bank_snapshot_is_current(state)
        and bool(state.get("virtual_response_data_ref"))
        and state.get("virtual_response_item_bank_id")
        == state.get("item_bank_id")
        and state.get("virtual_response_item_bank_version")
        == state.get("item_bank_version")
    )
    retention_gaps = build_blueprint_retention_gaps(state)
    pending_psychometric_repair = _next_psychometric_repair(state)
    if (
        pending_psychometric_repair is not None
        and not isinstance(
            pending_psychometric_repair.get("atomic_repair_advice"),
            Mapping,
        )
        and pending_psychometric_repair.get("queue_status")
        != "pending_diagnosis"
    ):
        raise ValueError("心理测量返修任务缺少结构化修改意见或诊断证据")
    blueprint = state.get("blueprint")
    pending_cell = state.get("current_blueprint_cell")
    pending_cell_id = (
        pending_cell.get("cell_id")
        if isinstance(pending_cell, Mapping)
        and state.get("current_item") is None
        else None
    )
    pending_replacement_cell = (
        dict(pending_cell)
        if isinstance(pending_cell_id, str)
        and isinstance(blueprint, dict)
        and pending_cell_id
        in {
            cell.get("cell_id")
            for cell in blueprint.get("cells") or []
            if isinstance(cell, Mapping)
        }
        and state["blueprint_progress"].get(pending_cell_id, {}).get(
            "missing", 0
        )
        > 0
        else None
    )
    selected_cell = None
    if isinstance(blueprint, dict):
        selected_cell = pending_replacement_cell or select_next_blueprint_cell(
            blueprint,
            state["blueprint_progress"],
        )

    if not state.get("requirements_confirmed", False):
        raw_decision = {
            "next_action": "clarify_requirements",
            "reason": "测验需求尚未经过用户最终确认",
            "target_item_id": None,
            "target_blueprint_cell_id": None,
        }
    elif state.get("blueprint") is None:
        raw_decision = {
            "next_action": "build_blueprint",
            "reason": (
                "从版本化构念库解析目标，合成情境、建立固定槽位与"
                "心理骨架并完成独立审核"
            ),
            "target_item_id": None,
            "target_blueprint_cell_id": None,
        }
    elif not _blueprint_review_allows_item_generation(state):
        raw_decision = {
            "next_action": "build_blueprint",
            "reason": "细目表尚未通过独立专家审核，禁止进入出题",
            "target_item_id": None,
            "target_blueprint_cell_id": None,
        }
    elif (
        isinstance(state.get("active_psychometric_repair"), Mapping)
        and state["active_psychometric_repair"].get("manual_edit_pending_review")
        and isinstance(state.get("current_item"), Mapping)
        and state.get("current_item_review") is None
    ):
        raw_decision = {
            "next_action": "review_item",
            "reason": "人工修改已通过本地字段锁定校验，进入完整内容审查",
            "target_item_id": state["current_item"].get("item_id"),
            "target_blueprint_cell_id": state["current_item"].get(
                "blueprint_cell_id"
            ),
        }
    elif _psychometric_confirmation_is_pending(state):
        raw_decision = {
            "next_action": "confirm_psychometric_repair",
            "reason": "等待用户确认当前单题心理测量诊断后再执行修改",
            "target_item_id": pending_psychometric_repair["item_id"]
            if pending_psychometric_repair
            else None,
            "target_blueprint_cell_id": (
                pending_psychometric_repair.get("blueprint_cell_id")
                if pending_psychometric_repair
                else None
            ),
        }
    elif (
        _psychometric_confirmation_is_approved(state)
        and pending_psychometric_repair is not None
    ):
        raw_decision = {
            # Psychometric repairs are always atomic patches. Even legacy
            # queue entries marked regenerate_item must use the scoped repair
            # agent so the fixed slot and diagnosis scope are preserved.
            "next_action": (
                "psychometric_repair_batch"
                if _has_batchable_psychometric_repairs(state)
                else "revise_item"
            ),
            "reason": (
                "当前批次诊断已确认，启动并发单题修改—复测闭环"
                if _has_batchable_psychometric_repairs(state)
                else "用户已确认当前单题诊断，进入原子修改"
            ),
            "target_item_id": pending_psychometric_repair["item_id"],
            "target_blueprint_cell_id": pending_psychometric_repair.get(
                "blueprint_cell_id"
            ),
        }
    elif (
        pending_replacement_cell is not None
    ):
        raw_decision = {
            "next_action": "generate_item",
            "reason": "先完成当前淘汰题的同槽位补题，再继续处理本轮其余未通过题",
            "target_item_id": None,
            "target_blueprint_cell_id": pending_replacement_cell.get("cell_id"),
        }
    elif pending_psychometric_repair is not None:
        if (
            not isinstance(
                pending_psychometric_repair.get("atomic_repair_advice"),
                Mapping,
            )
            and pending_psychometric_repair.get("queue_status")
            != "pending_diagnosis"
        ):
            raise ValueError("心理测量返修任务缺少结构化修改意见或诊断证据")
        raw_decision = {
            "next_action": "select_items",
            "reason": (
                "沿用本轮基线指标，继续诊断外层待处理队列中的下一道异常题；"
                "队列清空后再统一重新施测"
                if not isinstance(
                    pending_psychometric_repair.get("atomic_repair_advice"),
                    Mapping,
                )
                else "当前待修题尚未形成用户确认记录，重新进入单题诊断；"
                "整批处理完成后再统一重新施测"
            ),
            "target_item_id": None,
            "target_blueprint_cell_id": None,
        }
    elif (
        responses_bound_to_current_bank
        and not psychometrics_complete_before_route
    ):
        raw_decision = {
            "next_action": "analyze_psychometrics",
            "reason": "虚拟作答已完成，执行确定性心理测量分析",
            "target_item_id": None,
            "target_blueprint_cell_id": None,
        }
    elif (
        psychometrics_complete_before_route
        and state.get("selection_results") is None
    ):
        raw_decision = {
            "next_action": "select_items",
            "reason": "心理测量分析已完成，所有未通过题目进入返修诊断",
            "target_item_id": None,
            "target_blueprint_cell_id": None,
        }
    elif selection_status == "awaiting_sme_review":
        return {
            "status": "stopped",
            "route": {
                "next_action": "finish",
                "reason": "待SME题造成蓝图缺口，暂停组卷并保留检查点",
                "target_item_id": None,
                "target_blueprint_cell_id": None,
            },
            "execution_history": [
                *state.get("execution_history", []),
                {
                    "event_id": (
                        f'{state.get("run_id", "unknown")}:'
                        f'{state.get("step_count", 0)}:awaiting_sme_review'
                    ),
                    "run_id": state.get("run_id"),
                    "step": state.get("step_count", 0),
                    "node": "router",
                    "action": "awaiting_sme_review",
                    "event_type": "paused",
                    "recorded_at": utc_timestamp(),
                    "reason": "待SME题不进入正式题集合，当前蓝图不足以组卷",
                },
            ],
        }
    elif (
        selection_status == "fixed_blueprint_gap"
        and bool(retention_gaps)
    ):
        return _fixed_blueprint_gap_failure(
            state,
            reason="题目筛选后仍存在蓝图保留量缺口",
        )
    elif (
        selection_status == "fixed_blueprint_gap"
        and not retention_gaps
    ):
        raw_decision = {
            "next_action": "select_items",
            "reason": (
                "题目筛选结果报告固定蓝图缺口，但当前已无硬性保留量缺口；"
                "重新计算筛选结果，避免创建空补题请求"
            ),
            "target_item_id": None,
            "target_blueprint_cell_id": None,
        }
    elif (
        selection_status == "ready_for_assembly"
        and state.get("assembled_test") is None
    ):
        raw_decision = {
            "next_action": "assemble_test",
            "reason": "筛选后的题目已满足蓝图，按既定质量顺序组卷",
            "target_item_id": None,
            "target_blueprint_cell_id": None,
        }
    elif (
        state.get("assembled_test") is not None
        and test_review_decision_before_route is None
    ):
        raw_decision = {
            "next_action": "review_test",
            "reason": "正式测验已组卷，执行确定性测验级审核",
            "target_item_id": None,
            "target_blueprint_cell_id": None,
        }
    elif test_review_decision_before_route == "REASSEMBLE":
        raw_decision = {
            "next_action": "assemble_test",
            "reason": "测验级审核要求调整题目顺序并重新组卷",
            "target_item_id": None,
            "target_blueprint_cell_id": None,
        }
    elif test_review_decision_before_route == "RESCORE":
        raw_decision = {
            "next_action": "rescore_test",
            "reason": "测验级审核发现评分结构问题，进入重计分",
            "target_item_id": None,
            "target_blueprint_cell_id": None,
        }
    elif test_review_decision_before_route == "SUPPLEMENT":
        return _fixed_blueprint_gap_failure(
            state,
            reason="测验级审核发现阻断性蓝图缺口",
        )
    elif (
        test_review_decision_before_route == "PASS"
        and not final_outputs_exist
    ):
        raw_decision = {
            "next_action": "generate_reports",
            "reason": "测验级审核通过，生成最终测验和开发报告",
            "target_item_id": None,
            "target_blueprint_cell_id": None,
        }
    elif (
        test_review_decision_before_route == "PASS"
        and final_outputs_exist
    ):
        completion_checks, unmet = evaluate_completion(state)
        raw_decision = {
            # Always pass through the programmatic finish gate below. It either
            # finishes or selects a safe upstream recovery action.
            "next_action": "finish",
            "reason": (
                "全部程序化完成条件均已满足"
                if not unmet
                else "请求完成条件门禁处理尚未满足的条件："
                + "、".join(unmet)
            ),
            "target_item_id": None,
            "target_blueprint_cell_id": None,
        }
    elif selected_cell is not None:
        raw_decision = {
            "next_action": "generate_item",
            "reason": (
                "蓝图仍有未尝试的题目生成槽"
            ),
            "target_item_id": None,
            "target_blueprint_cell_id": None,
        }
    elif retention_gaps:
        return _fixed_blueprint_gap_failure(
            state,
            reason="程序检查发现固定细目表仍有保留量硬缺口",
        )
    elif not responses_bound_to_current_bank:
        raw_decision = {
            "next_action": "simulate_responses",
            "reason": (
                "候选题已完成逐题审查且蓝图保留量无缺口；"
                "程序冻结当前版本并进入虚拟施测"
            ),
            "target_item_id": None,
            "target_blueprint_cell_id": None,
        }
    else:
        raise ValueError(
            "当前 State 无法确定下一步动作："
            f"selection_status={selection_status!r}, "
            f"test_review_decision={test_review_decision_before_route!r}"
        )

    decision = {
        "next_action": raw_decision["next_action"],
        "reason": raw_decision["reason"],
        "target_item_id": raw_decision.get("target_item_id"),
        "target_blueprint_cell_id": raw_decision.get(
            "target_blueprint_cell_id"
        ),
    }
    route_update: dict[str, Any] = {}
    if state.get("skeleton_slot_failure_pending"):
        route_update["skeleton_slot_failure_pending"] = False
    if (
        pending_psychometric_repair is not None
        and decision["next_action"] in {"revise_item", "regenerate_item"}
    ):
        route_update.update(
            _stage_psychometric_repair(
                state,
                pending_psychometric_repair,
            )
        )
    if decision["next_action"] == "generate_item":
        if not isinstance(blueprint, dict):
            decision = {
                "next_action": "build_blueprint",
                "reason": "生成题目前缺少有效蓝图，返回蓝图设计阶段",
                "target_item_id": None,
                "target_blueprint_cell_id": None,
            }
        elif not _blueprint_review_allows_item_generation(state):
            decision = {
                "next_action": "build_blueprint",
                "reason": "独立细目表审核未 PASS，程序禁止进入出题",
                "target_item_id": None,
                "target_blueprint_cell_id": None,
            }
        else:
            if selected_cell is None:
                if retention_gaps:
                    return _fixed_blueprint_gap_failure(
                        state,
                        reason="固定细目表全部槽位已尝试但保留量仍不足",
                    )
                decision = {
                    "next_action": "simulate_responses",
                    "reason": "候选题已完成，冻结当前版本并进入虚拟施测",
                    "target_item_id": None,
                    "target_blueprint_cell_id": None,
                }
                route_update["current_blueprint_cell"] = None
                route_update["current_item_specification"] = None
            else:
                decision["target_item_id"] = None
                decision["target_blueprint_cell_id"] = selected_cell["cell_id"]
                route_update["current_blueprint_cell"] = selected_cell
                generated_count = state["blueprint_progress"].get(
                    selected_cell["cell_id"], {}
                ).get("generated", 0)
                item_specification = select_item_specification(
                    state.get("item_specifications", []),
                    selected_cell["cell_id"],
                    generated_count,
                )
                if item_specification is None:
                    cell_slots = [
                        slot
                        for slot in blueprint.get("slots") or []
                        if isinstance(slot, Mapping)
                        and slot.get("blueprint_cell_id")
                        == selected_cell["cell_id"]
                    ]
                    if generated_count >= len(cell_slots):
                        raise ValueError(
                            f"蓝图单元 {selected_cell['cell_id']} 缺少固定题目槽"
                        )
                    slot = cell_slots[generated_count]
                    item_specification = {
                        "specification_id": slot["specification_id"],
                        "blueprint_cell_id": slot["blueprint_cell_id"],
                        "target_dimension_id": selected_cell["facet_id"],
                    }
                route_update["current_item_specification"] = item_specification
    if (
        decision["next_action"] == "simulate_responses"
        and selected_cell is None
    ):
        route_update["current_blueprint_cell"] = None
        route_update["current_item_specification"] = None
    if decision["next_action"] == "finish":
        checks, unmet = evaluate_completion(state)
        route_update["completion_checks"] = checks
        route_update["unmet_completion_conditions"] = unmet
        if unmet:
            if (
                not checks["item_bank_current"]
                or not checks["responses_bound_to_current_bank"]
            ):
                guarded_action = "simulate_responses"
            elif not checks["psychometrics_complete"]:
                guarded_action = "analyze_psychometrics"
            elif not checks["selection_ready"]:
                guarded_action = "select_items"
            elif not checks["blueprint_coverage_passed"]:
                return _fixed_blueprint_gap_failure(
                    state,
                    reason="完成条件检查发现固定蓝图覆盖不足",
                )
            elif not checks["assembly_matches_current_bank"]:
                guarded_action = "assemble_test"
            elif not checks["test_review_passed"]:
                review = state.get("test_review_result")
                review_decision = (
                    review.get("decision")
                    if isinstance(review, Mapping)
                    else None
                )
                guarded_action = {
                    "REASSEMBLE": "assemble_test",
                    "RESCORE": "rescore_test",
                }.get(review_decision, "review_test")
            elif not (
                checks["final_test_exists"]
                and checks["item_database_exists"]
                and checks["technical_report_exists"]
                and checks["virtual_respondent_report_exists"]
            ):
                guarded_action = "generate_reports"
            elif not checks["no_pending_rescore_revalidation"]:
                guarded_action = "simulate_responses"
            elif state.get("test_review_result") is None:
                guarded_action = "review_test"
            else:
                raise ValueError(
                    "finish 被程序门禁阻止，且当前状态没有安全的后续动作："
                    + "、".join(unmet)
                )
            decision = {
                "next_action": guarded_action,
                "reason": "finish 被程序化完成条件阻止：" + "、".join(unmet),
                "target_item_id": None,
                "target_blueprint_cell_id": None,
            }
    is_finished = decision["next_action"] == "finish"
    if is_finished:
        route_update["current_phase"] = "completed"

    return {
        "route": decision,
        **route_update,
        "status": "completed" if is_finished else state["status"],
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": f'{state["run_id"]}:{state["step_count"]}:router:completed',
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "router",
                "action": decision["next_action"],
                "event_type": "completed",
                "reason": decision["reason"],
                "recorded_at": utc_timestamp(),
                "duration_ms": round((perf_counter() - started_at) * 1000),
            },
        ],
    }
