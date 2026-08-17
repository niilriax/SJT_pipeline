"""User interaction, approval, commit, regeneration, and stop nodes."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from langgraph.types import interrupt

from sjt_system.authoring.blueprint import (
    initialize_blueprint_progress,
    validate_blueprint_agent_update,
    validate_integrated_blueprint,
)
from sjt_system.authoring.requirements import validate_requirement_confirmation
from sjt_system.evaluation.respondents import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_SELECTION_SEED,
    MAX_ALLOWED_CONCURRENCY,
    build_virtual_pool_summary,
    build_virtual_sample_config,
    build_virtual_sample_recommendations,
    load_virtual_respondent_pool,
    select_virtual_respondent_refs,
)
from sjt_system.evaluation.diagnosis import (
    item_requires_psychometric_diagnosis,
)
from sjt_system.runtime.trace import build_state_diff, utc_timestamp
from sjt_system.state import PSJTState
from sjt_system.workflow.approval import (
    build_committed_update,
    normalize_user_decision,
)
from sjt_system.workflow.constants import (
    DETERMINISTIC_AUTO_APPROVAL_ACTIONS,
    ITEM_DEVELOPMENT_ACTIONS,
    PHASE_BY_ACTION,
)

def approval_node(state: PSJTState) -> dict:
    """暂停工作流，等待用户处理本次 Agent 的候选结果。"""

    pending_update = state.get("pending_state_update")
    if pending_update is None:
        raise ValueError("Approval 节点没有收到待确认的 Agent 结果")

    is_requirement_action = (
        state.get("pending_action") == "clarify_requirements"
    )
    requirement_errors = (
        validate_requirement_confirmation(
            pending_update.get("test_specification"),
            state.get("pending_interaction"),
            state["confirmed_requirement_fields"],
            pending_update.get("specification_sources"),
        )
        if is_requirement_action
        else {}
    )
    inferred_fields = {
        field
        for field, source in (
            pending_update.get("specification_sources") or {}
        ).items()
        if source == "inferred"
    }
    requirement_decisions = (
        ["confirm", "revise", "stop"]
        if not requirement_errors
        else [
            "answer",
            *(
                ["accept_suggestions"]
                if (
                    (state.get("pending_interaction") or {}).get("suggestions")
                    or inferred_fields
                )
                else []
            ),
            "stop",
        ]
    )
    payload: dict[str, Any] = {
        "type": (
            "requirement_confirmation"
            if is_requirement_action
            else "agent_result_approval"
        ),
        "action": state.get("pending_action"),
        "summary": state.get("pending_summary"),
        "proposed_update": pending_update,
        "state_changes": state.get("pending_state_changes") or {},
        "available_decisions": (
            requirement_decisions
            if is_requirement_action
            else ["approve", "edit", "regenerate", "stop"]
        ),
    }
    if is_requirement_action:
        payload.update(state.get("pending_interaction") or {})
        if requirement_errors:
            payload["field_errors"] = requirement_errors
    while True:
        raw_decision = interrupt(payload)
        try:
            decision = normalize_user_decision(raw_decision, pending_update)
            if is_requirement_action:
                if decision["decision"] not in {
                    "answer",
                    "accept_suggestions",
                    "confirm",
                    "revise",
                    "stop",
                }:
                    raise ValueError("需求阶段不支持该决策")
                if (
                    decision["decision"] == "accept_suggestions"
                    and not payload.get("suggestions")
                    and not inferred_fields
                ):
                    raise ValueError(
                        "当前没有可接受的系统建议或默认推断值"
                    )
                if decision["decision"] == "confirm":
                    errors = validate_requirement_confirmation(
                        pending_update.get("test_specification"),
                        state.get("pending_interaction"),
                        state["confirmed_requirement_fields"],
                        pending_update.get("specification_sources"),
                    )
                    if errors:
                        payload = {
                            **payload,
                            "validation_error": "候选需求尚不能最终确认",
                            "field_errors": errors,
                        }
                        continue
            elif decision["decision"] not in {
                "approve",
                "edit",
                "regenerate",
                "stop",
            }:
                raise ValueError("普通 Agent 审批不支持该决策")
            break
        except ValueError as exc:
            payload = {**payload, "validation_error": str(exc)}

    sourced_decision = {
        **decision,
        "approval_source": "user",
    }
    return {
        "user_decision": sourced_decision,
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": (
                    f'{state["run_id"]}:{state["step_count"]}:'
                    "approval:completed"
                ),
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "approval",
                "action": decision["decision"],
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": decision.get("feedback") or "用户已提交决策",
                "approval_source": "user",
            },
        ],
    }


def item_development_mode_selection_node(state: PSJTState) -> dict:
    """在首次执行题目开发动作前暂停并选择本次运行的开发模式。"""

    if state.get("item_development_mode") in {"manual", "automatic"}:
        return {}

    payload = {
        "type": "item_development_mode_selection",
        "modes": [
            {
                "mode": "manual",
                "label": "人工逐题",
                "description": "每次出题、审题、修改和重写后都由人工确认。",
            },
            {
                "mode": "automatic",
                "label": "Agent 自动完成",
                "description": (
                    "自动完成出题、审题、修改、重写、题库冻结、虚拟验证、"
                    "筛选和组卷；终端持续展示题目与进度。"
                ),
            },
        ],
    }
    while True:
        raw_selection = interrupt(payload)
        if isinstance(raw_selection, Mapping):
            mode = raw_selection.get("mode")
            if mode in {"manual", "automatic"}:
                break
        payload = {
            **payload,
            "validation_error": (
                "题目开发模式必须是 manual 或 automatic"
            ),
        }

    return {
        "item_development_mode": mode,
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": (
                    f'{state["run_id"]}:{state["step_count"]}:'
                    "item_development_mode:completed"
                ),
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "item_development_mode_selection",
                "action": mode,
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": (
                    "用户选择人工逐题模式"
                    if mode == "manual"
                    else "用户选择 Agent 自动完成模式"
                ),
                "approval_source": "user",
            },
        ],
    }


def virtual_sample_selection_node(state: PSJTState) -> dict:
    """在首次虚拟作答前让用户选择从内置被试池使用多少人。"""

    current_config = state.get("virtual_sample_config")
    current_respondents = state.get("virtual_respondents") or []
    if (
        isinstance(current_config, Mapping)
        and current_config.get("sample_size") == len(current_respondents)
        and current_respondents
    ):
        return {}

    pool = load_virtual_respondent_pool()
    summary = build_virtual_pool_summary(pool)
    recommendations = build_virtual_sample_recommendations(
        summary["available_count"]
    )
    payload = {
        "type": "virtual_sample_selection",
        "pool": summary,
        "recommendations": recommendations,
        "recommended_sample_size": next(
            option["sample_size"]
            for option in recommendations
            if option["recommended"]
        ),
        "default_seed": DEFAULT_SELECTION_SEED,
        "default_max_concurrency": DEFAULT_MAX_CONCURRENCY,
        "max_allowed_concurrency": MAX_ALLOWED_CONCURRENCY,
        "default_max_retries": DEFAULT_MAX_RETRIES,
        "method_note": (
            "同一被试默认分别使用“逐题作答”与“人格总结＋逐题作答”"
            "两种提示完成同一套测验；人格总结仅由39项人格作答生成一次并缓存。"
            "SJT按单题独立调用，Neo-FFI按五个12题批次调用；"
            "所有请求进入同一个受控并发队列。"
        ),
    }

    while True:
        raw_selection = interrupt(payload)
        try:
            if not isinstance(raw_selection, Mapping):
                raise ValueError("虚拟样本选择必须是对象")
            sample_size = raw_selection.get("sample_size")
            seed = raw_selection.get("seed", DEFAULT_SELECTION_SEED)
            max_concurrency = raw_selection.get(
                "max_concurrency",
                DEFAULT_MAX_CONCURRENCY,
            )
            max_retries = raw_selection.get(
                "max_retries",
                DEFAULT_MAX_RETRIES,
            )
            if isinstance(sample_size, str) and sample_size.isdigit():
                sample_size = int(sample_size)
            if isinstance(seed, str) and seed.lstrip("-").isdigit():
                seed = int(seed)
            if (
                isinstance(max_concurrency, str)
                and max_concurrency.isdigit()
            ):
                max_concurrency = int(max_concurrency)
            if isinstance(max_retries, str) and max_retries.isdigit():
                max_retries = int(max_retries)
            selected = select_virtual_respondent_refs(
                pool,
                sample_size,
                seed=seed,
            )
            config = build_virtual_sample_config(
                pool,
                sample_size,
                seed=seed,
                max_concurrency=max_concurrency,
                max_retries=max_retries,
            )
            break
        except (TypeError, ValueError) as exc:
            payload = {**payload, "validation_error": str(exc)}

    return {
        "virtual_sample_config": config,
        "virtual_respondents": selected,
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": (
                    f'{state["run_id"]}:{state["step_count"]}:'
                    "virtual_sample_selection:completed"
                ),
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "virtual_sample_selection",
                "action": "configure_virtual_sample",
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": (
                    f"用户从 {summary['available_count']} 名匿名人格档案中"
                    f"选择 {sample_size} 名；随机种子 {seed}"
                    f"；最大并发 {max_concurrency}"
                ),
                "approval_source": "user",
            },
        ],
    }


def post_virtual_response_decision_node(state: PSJTState) -> dict:
    """Ask once whether the user wants the psychometric repair pass enabled."""

    summary = state.get("virtual_response_summary") or {}
    payload = {
        "type": "post_virtual_response_decision",
        "summary": (
            "虚拟被试作答已完成。是否开始根据本轮心理测量结果修改题目？"
        ),
        "virtual_response_summary": summary,
        "available_decisions": ["start", "skip", "stop"],
    }
    while True:
        raw_decision = interrupt(payload)
        decision = raw_decision.get("decision") if isinstance(raw_decision, Mapping) else None
        if decision in {"start", "skip", "stop"}:
            break
        payload = {**payload, "validation_error": "请输入 1、2 或 3"}
    if decision == "stop":
        return {"status": "stopped"}
    update = {
        "psychometric_repair_user_decision": decision,
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": f'{state["run_id"]}:{state["step_count"]}:post_virtual_response_decision:completed',
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "post_virtual_response_decision",
                "action": decision,
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": "用户选择是否开始心理测量修题",
                "approval_source": "user",
            },
        ],
    }
    if decision == "skip":
        # No repair pass: the current structurally valid item pool goes
        # directly to assembly. There is no selection/defer stage anymore.
        current_items = [
            dict(item)
            for item in state.get("item_pool") or []
            if isinstance(item, Mapping)
        ]
        item_counts: dict[str, int] = {}
        for item in current_items:
            cell_id = item.get("blueprint_cell_id")
            if isinstance(cell_id, str):
                item_counts[cell_id] = item_counts.get(cell_id, 0) + 1
        coverage_cells = []
        coverage_passed = True
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
        final_dispositions = {
            str(item.get("item_id")): {
                "status": (
                    "accepted_with_warning"
                    if item_requires_psychometric_diagnosis(
                        (state.get("item_statistics") or {}).get(item.get("item_id")) or {}
                    )
                    else "accepted"
                ),
                "warning_reason": (
                    "user_skipped_repair"
                    if item_requires_psychometric_diagnosis(
                        (state.get("item_statistics") or {}).get(item.get("item_id")) or {}
                    )
                    else None
                ),
                "item_version": item.get("version"),
            }
            for item in current_items
            if item.get("item_id")
        }
        update.update(
            {
                "selected_items": current_items,
                "reserve_items": [],
                "blueprint_coverage": {
                    "passed": coverage_passed,
                    "cells": coverage_cells,
                    "expected_total": sum(
                        cell["planned_retention_count"]
                        for cell in coverage_cells
                    ),
                    "selected_total": len(current_items),
                },
                "selection_results": {
                    "status": (
                        "ready_for_assembly"
                        if coverage_passed
                        else "fixed_blueprint_gap"
                    ),
                    "direct_repair": True,
                    "repair_count": 0,
                    "selected_count": len(state.get("item_pool") or []),
                    "reserve_count": 0,
                    "final_dispositions": final_dispositions,
                },
                "item_final_dispositions": final_dispositions,
            }
        )
    return update


def psychometric_repair_confirmation_node(state: PSJTState) -> dict:
    """Ask whether to apply the one diagnosis currently shown to the user."""

    pending = state.get("psychometric_repair_confirmation")
    if not isinstance(pending, Mapping) or pending.get("status") != "pending":
        raise ValueError("当前没有等待确认的单题心理测量返修建议")
    advice = pending.get("atomic_repair_advice") or {}
    evidence = pending.get("diagnosis_evidence") or {}
    item = evidence.get("current_item") or {}
    payload = {
        "type": "psychometric_repair_confirmation",
        "item_id": pending.get("item_id"),
        "revision_round": pending.get("revision_round"),
        "pending_item_queue": [
            {
                "item_id": entry.get("item_id"),
                "revision_round": entry.get("revision_round"),
                "queue_status": entry.get("queue_status", "pending_diagnosis"),
            }
            for entry in state.get("items_to_revise") or []
            if isinstance(entry, Mapping) and entry.get("item_id")
        ],
        "item": item,
        "diagnosis": {
            "summary": advice.get("summary"),
            "decision": advice.get("decision"),
            "observed_discrepancies": advice.get("observed_discrepancies") or [],
            "candidate_diagnoses": advice.get("candidate_diagnoses") or [],
            "repair_tasks": advice.get("repair_tasks") or [],
            "selected_diagnosis_id": advice.get("selected_diagnosis_id"),
            "atomic_edit": advice.get("atomic_edit"),
        },
        "observations": evidence.get("observations") or [],
        "option_evidence": evidence.get("option_evidence") or [],
        "available_decisions": ["approve", "skip", "stop"],
        "instruction": (
            "approve：按当前题目的全部已确认问题逐条调用修题模型，完成后统一重新施测；"
            "skip：不修改本题并记录警告，继续下一题；"
            "stop：暂停本次运行。"
        ),
    }
    while True:
        raw = interrupt(payload)
        decision = raw.get("decision") if isinstance(raw, Mapping) else None
        if decision in {"approve", "skip", "stop"}:
            break
        payload = {**payload, "validation_error": "decision 必须是 approve、skip 或 stop"}
    if decision == "stop":
        return {"status": "stopped"}

    history = [
        *state.get("execution_history", []),
        {
            "event_id": (
                f'{state.get("run_id", "unknown")}:{state.get("step_count", 0)}:'
                "psychometric_repair_confirmation:completed"
            ),
            "run_id": state.get("run_id"),
            "step": state.get("step_count", 0),
            "node": "psychometric_repair_confirmation",
            "action": decision,
            "event_type": "completed",
            "recorded_at": utc_timestamp(),
            "reason": "用户确认当前单题心理测量返修建议",
            "approval_source": "user",
        },
    ]
    if decision == "approve":
        return {
            "psychometric_repair_confirmation": {
                **dict(pending),
                "status": "approved",
                "decision": "approve",
            },
            "execution_history": history,
        }

    item_id = str(pending.get("item_id") or "")
    dispositions = deepcopy(state.get("item_final_dispositions") or {})
    item_version = (item or {}).get("version")
    dispositions[item_id] = {
        "status": "accepted_with_warning",
        "warning_reason": "user_skipped_repair",
        "item_version": item_version,
    }
    remaining = [
        deepcopy(entry)
        for entry in state.get("items_to_revise") or []
        if isinstance(entry, Mapping) and entry.get("item_id") != item_id
    ]
    remaining_regenerate = [
        deepcopy(entry)
        for entry in state.get("items_to_regenerate") or []
        if isinstance(entry, Mapping) and entry.get("item_id") != item_id
    ]
    return {
        "psychometric_repair_confirmation": None,
        "active_psychometric_repair": None,
        "items_to_revise": remaining,
        "items_to_regenerate": remaining_regenerate,
        "item_final_dispositions": dispositions,
        "selection_reasons": {
            **deepcopy(state.get("selection_reasons") or {}),
            item_id: "用户跳过当前单题返修，保留当前版本并记录警告",
        },
        "psychometric_repair_history": [
            *deepcopy(state.get("psychometric_repair_history") or []),
            {
                "event": "psychometric_item_repair_skipped",
                "recorded_at": utc_timestamp(),
                "item_id": item_id,
                "revision_round": pending.get("revision_round"),
                "diagnosis_fingerprint": pending.get("diagnosis_fingerprint"),
                "reason": "user_skipped_repair",
            },
        ],
        "selection_results": None,
        "execution_history": history,
    }


def automatic_approval_node(state: PSJTState) -> dict:
    """为自动模式下已通过执行校验的题目动作生成系统批准决策。"""

    action = state.get("pending_action")
    is_automatic_item_action = (
        state.get("item_development_mode") == "automatic"
        and action in ITEM_DEVELOPMENT_ACTIONS
    )
    is_automatic_deterministic_action = (
        state.get("item_development_mode") == "automatic"
        and action in DETERMINISTIC_AUTO_APPROVAL_ACTIONS
    )
    if (
        not is_automatic_item_action
        and not is_automatic_deterministic_action
    ):
        raise ValueError(
            "自动审批仅适用于自动模式下的闭环动作"
        )
    if state.get("pending_state_update") is None:
        raise ValueError("自动审批节点没有收到待提交的 Agent 结果")

    decision = {
        "decision": "approve",
        "feedback": None,
        "state_patch": None,
        "approval_source": "system",
    }
    return {
        "user_decision": decision,
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": (
                    f'{state["run_id"]}:{state["step_count"]}:'
                    "approval:completed"
                ),
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "approval",
                "action": "approve",
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": f"自动模式已批准题目动作 {action}",
                "approval_source": "system",
            },
        ],
    }


def commit_node(state: PSJTState) -> dict:
    """把用户确认后的候选更新写入正式业务 State。"""

    pending_update = state.get("pending_state_update")
    decision = state.get("user_decision")
    if pending_update is None or decision is None:
        raise ValueError("Commit 节点缺少候选结果或用户决策")

    committed_update = build_committed_update(pending_update, decision)
    action = state.get("pending_action") or "unknown"
    requirement_update: dict[str, Any] = {}
    if action == "clarify_requirements":
        if decision["decision"] != "confirm":
            raise ValueError("只有 confirm 决策可以提交需求规格")
        errors = validate_requirement_confirmation(
            committed_update.get("test_specification"),
            state.get("pending_interaction"),
            state["confirmed_requirement_fields"],
            committed_update.get("specification_sources"),
        )
        if errors:
            raise ValueError(f"需求确认校验失败：{errors}")
        requirement_update["requirements_confirmed"] = True
    blueprint_update: dict[str, Any] = {}
    if action == "build_blueprint":
        validate_blueprint_agent_update(committed_update)
        blueprint = committed_update.get("blueprint")
        errors = validate_integrated_blueprint(
            blueprint,
            state.get("test_specification"),
        )
        if errors:
            raise ValueError(f"蓝图提交校验失败：{errors}")
        blueprint_update["blueprint_progress"] = (
            initialize_blueprint_progress(blueprint)
        )
        if "construct_profile_snapshot" in blueprint:
            profile = blueprint["construct_profile_snapshot"]
            blueprint_update["construct_profile"] = profile
            blueprint_update["item_skeletons"] = {}
            blueprint_update["skeleton_reviews"] = {}
            blueprint_update["item_specifications"] = []
        else:
            raise ValueError("当前工作流不再接受旧版动态构念蓝图")
    full_committed_update = {
        **committed_update,
        **requirement_update,
        **blueprint_update,
        "current_phase": PHASE_BY_ACTION.get(
            action,
            state.get("current_phase"),
        ),
    }
    state_changes = build_state_diff(state, full_committed_update)
    approval_source = decision.get("approval_source", "user")

    return {
        **full_committed_update,
        "pending_action": None,
        "pending_state_update": None,
        "pending_summary": None,
        "pending_state_changes": None,
        "pending_interaction": None,
        "user_decision": None,
        "user_feedback": None,
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": (
                    f'{state["run_id"]}:{state["step_count"]}:commit:completed'
                ),
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "commit",
                "action": action,
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": (
                    "系统已自动批准，候选结果写入正式 State"
                    if approval_source == "system"
                    else "用户已确认，候选结果写入正式 State"
                ),
                "state_changes": state_changes,
                "approval_source": approval_source,
            },
        ],
    }


def prepare_regeneration_node(state: PSJTState) -> dict:
    """丢弃候选结果，并保留用户意见供同一 Agent 重新生成。"""

    decision = state.get("user_decision") or {}
    action = state.get("pending_action") or "unknown"
    is_requirement_action = action == "clarify_requirements"
    feedback = decision.get("feedback")
    confirmed_fields = state["confirmed_requirement_fields"]
    if is_requirement_action and decision.get("decision") == "accept_suggestions":
        suggestion_fields = {
            item.get("field")
            for item in (
                (state.get("pending_interaction") or {}).get("suggestions")
                or []
            )
            if isinstance(item, dict) and isinstance(item.get("field"), str)
        }
        pending_sources = (
            (state.get("pending_state_update") or {}).get(
                "specification_sources",
                {},
            )
        )
        inferred_fields = {
            field
            for field, source in pending_sources.items()
            if source == "inferred"
        }
        accepted_fields = suggestion_fields | inferred_fields
        confirmed_fields = sorted({*confirmed_fields, *accepted_fields})
        feedback = (
            "我接受本轮全部系统建议：" + "、".join(sorted(accepted_fields))
        )
    conversation = state["requirement_conversation"]
    if is_requirement_action and feedback:
        conversation = [
            *conversation,
            {"role": "user", "content": feedback},
        ]
    return {
        "pending_action": None,
        # Requirement Agent 需要看到上一版候选，才能只修改用户指出的字段。
        "pending_state_update": (
            state.get("pending_state_update")
            if is_requirement_action
            else None
        ),
        "pending_summary": None,
        "pending_state_changes": None,
        "pending_interaction": None,
        "user_decision": None,
        "user_feedback": feedback,
        "confirmed_requirement_fields": confirmed_fields,
        "requirement_conversation": conversation,
        "requirements_confirmed": (
            False
            if is_requirement_action
            else state["requirements_confirmed"]
        ),
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": (
                    f'{state["run_id"]}:{state["step_count"]}:'
                    "regenerate:completed"
                ),
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "prepare_regeneration",
                "action": action,
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": feedback or "用户要求继续需求澄清",
            },
        ],
    }


def stop_node(state: PSJTState) -> dict:
    """记录用户主动停止并结束工作流。"""

    decision = state.get("user_decision") or {}
    action = state.get("pending_action") or "unknown"
    return {
        "status": "stopped",
        "pending_action": None,
        "pending_state_update": None,
        "pending_summary": None,
        "pending_state_changes": None,
        "pending_interaction": None,
        "user_decision": None,
        "user_feedback": decision.get("feedback"),
        "execution_history": [
            *state["execution_history"],
            {
                "event_id": f'{state["run_id"]}:{state["step_count"]}:stop:completed',
                "run_id": state["run_id"],
                "step": state["step_count"],
                "node": "stop",
                "action": action,
                "event_type": "completed",
                "recorded_at": utc_timestamp(),
                "reason": decision.get("feedback") or "用户主动停止工作流",
            },
        ],
    }
