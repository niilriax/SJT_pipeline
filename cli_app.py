"""Legacy command-line interface for the SJT workflow."""

import asyncio
import os
from pathlib import Path
from pprint import pprint
from time import monotonic
from typing import Callable

from langgraph.types import Command

from sjt_system.workflow.graph import build_sjt_graph
from sjt_system.runtime.progress import progress_callback
from sjt_system.runtime.checkpoint import (
    DEFAULT_CHECKPOINT_ROOT,
    find_latest_resumable_checkpoint,
    prepare_retry_state,
    prepare_resumed_state,
    save_run_checkpoint,
)
from sjt_system.state import TraceEvent, create_initial_state
from sjt_system.authoring.items import derive_item_review_decision
from sjt_system.authoring.construct_registry import (
    resolve_specification_profile,
)
from sjt_system.authoring.generation_plan import (
    planned_generation_count,
    planned_retention_count,
)

app = build_sjt_graph()


SPECIFICATION_LABELS = {
    "construct_selection": "测量构念",
    "target_population": "目标人群",
    "final_item_count": "最终题量",
    "output_language": "输出语言",
}

ACTION_LABELS = {
    "clarify_requirements": "需求澄清",
    "build_blueprint": "构念—题目细目表",
    "generate_item": "生成题目",
    "review_item": "审查题目",
    "revise_item": "定向修改题目",
    "regenerate_item": "重写题目",
    "simulate_responses": "虚拟被试作答",
    "analyze_psychometrics": "心理测量分析",
    "select_items": "心理测量返修诊断",
    "assemble_test": "测验组卷",
    "review_test": "测验整体审核",
    "rescore_test": "重新计分",
    "generate_reports": "生成测验与报告",
    "finish": "完成工作流",
}

ITEM_OUTPUT_ACTIONS = {
    "generate_item": "生成",
    "revise_item": "定向修改",
    "regenerate_item": "重写",
}

ACTION_START_NODES = {
    "router",
    "prepare_item_review",
    "prepare_item_revision",
}


def action_label(action: object) -> str:
    value = str(action or "unknown")
    return ACTION_LABELS.get(value, value)


def item_stage_label(action: str, state: dict) -> str:
    if action == "review_item":
        return (
            "复审题目"
            if state.get("current_item_repair_attempted")
            else "首次审题"
        )
    return action_label(action)


def print_runtime_progress(event: dict) -> None:
    """Render transient model and simulation progress events."""

    event_type = event.get("type")
    if event_type == "simulation_stage":
        status = {
            "started": "开始",
            "completed": "完成",
            "failed": "失败",
        }.get(event.get("status"), event.get("status"))
        print(
            f"\n[虚拟作答] {event.get('stage')} {status}"
            f"（本轮调用 {event.get('total', '?')} 次）",
            flush=True,
        )
    elif event_type == "simulation_progress":
        completed = event.get("completed", 0)
        total = event.get("total", 0)
        percent = 100 if total == 0 else round(completed / total * 100)
        print(
            f"[虚拟作答] {event.get('stage')}："
            f"{completed}/{total}（{percent}%）",
            flush=True,
        )
    elif event_type in {"request_retry", "output_repair"}:
        print(
            f"\n[重试] {event.get('job_label', '模型请求')}："
            f"第 {event.get('attempt', '?')}/"
            f"{event.get('max_attempts', '?')} 次；"
            f"原因：{event.get('reason', '未知')}",
            flush=True,
        )
    elif event_type == "action_fallback":
        print(
            f"\n[升级处理] {action_label(event.get('from_action'))}"
            f"连续 {event.get('failed_attempts', '?')} 次未通过校验，"
            f"已自动转为{action_label(event.get('to_action'))}；"
            f"原因：{event.get('reason', '未知')}",
            flush=True,
        )
    elif event_type == "request_timeout":
        print(
            f"\n[超时] {event.get('job_label', '模型请求')} 超过 "
            f"{event.get('timeout_seconds', '?')} 秒，已取消。",
            flush=True,
        )
    elif event_type == "skeleton_slot_failed":
        print(
            "\n[骨架槽位失败] 当前骨架未通过程序确定性校验，当前槽位将被拒绝，"
            "工作流继续处理下一固定槽位。",
            flush=True,
        )
        if event.get("final_reason"):
            print(f"最终原因：{event['final_reason']}", flush=True)


def print_automatic_item_result(
    action: str,
    update: dict,
    state: dict,
) -> None:
    """Show automatic-mode item outputs without adding approval pauses."""

    if state.get("item_development_mode") != "automatic":
        return
    proposed_update = update.get("pending_state_update") or {}
    if action in ITEM_OUTPUT_ACTIONS:
        print_skeleton_candidate(proposed_update, state)
        print(f"\n===== 题目已{ITEM_OUTPUT_ACTIONS[action]} =====")
        print_item_candidate(proposed_update.get("current_item"))
    elif action == "review_item":
        print("\n===== 题目审查结果 =====")
        print_review_candidate(proposed_update)


def _format_metric(value: object, *, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "证据不足"
    return f"{float(value):.{digits}f}"


def print_psychometric_summary(
    proposed_update: dict,
    state: dict,
) -> None:
    statistics = proposed_update.get("test_statistics") or {}
    evaluation = statistics.get("measurement_evaluation") or {}
    reliability = evaluation.get("reliability") or {}
    validity = evaluation.get("validity") or {}
    convergent = validity.get("convergent") or {}
    discriminant = validity.get("discriminant") or {}
    counts = evaluation.get("item_recommendation_counts") or {}
    print("\n===== 心理测量核心结果 =====")
    print(
        f"样本量：{(state.get('virtual_sample_config') or {}).get('sample_size', '未知')}"
    )
    print(
        "信度："
        f"Cronbach α={_format_metric(statistics.get('cronbach_alpha'))}；"
        f"等级={reliability.get('overall_grade', '证据不足')}"
    )
    print(
        "效度："
        f"目标 {convergent.get('target', '?')} 相关="
        f"{_format_metric(convergent.get('rho'))}；"
        f"最大非目标维度={discriminant.get('largest_non_target_dimension', '?')}；"
        f"最大非目标相关="
        f"{_format_metric(discriminant.get('largest_non_target_rho'))}；"
        f"区分差值={_format_metric(discriminant.get('target_margin'))}"
    )
    print(
        "题目建议："
        f"retain={counts.get('retain', 0)}，"
        f"revise={counts.get('revise', 0)}，"
        f"remove={counts.get('remove', 0)}"
    )
    item_statistics = proposed_update.get("item_statistics") or {}
    dimensions = statistics.get("dimensions") or {}
    dimension_ids = list(
        dict.fromkeys(
            str(item.get("dimension_id") or "unknown")
            for item in item_statistics.values()
        )
    )
    for dimension_id in dimension_ids:
        dimension = dimensions.get(dimension_id) or {}
        print(
            f"\nFacet {dimension_id} 题项测量结果："
            f"Cronbach α={_format_metric(dimension.get('cronbach_alpha'))}"
        )
        print("| 题目 | 分面内CITC | 难度 | 有效选项数 | 结果 |")
        print("|---|---:|---:|---:|---|")
        for item_id, item in item_statistics.items():
            if str(item.get("dimension_id") or "unknown") != dimension_id:
                continue
            quality = item.get("quality_evaluation") or {}
            corrected = (
                item.get("facet_corrected_item_total_correlation") or {}
            )
            flags = "；".join(quality.get("diagnostic_flags") or []) or "无"
            row_label = item_id.split("-row-")[-1].split("-")[0]
            print(
                f"| Row {row_label} | {_format_metric(corrected.get('r'))} | "
                f"{_format_metric(item.get('difficulty'))} | "
                f"{quality.get('effective_option_count', '?')} | "
                f"{quality.get('recommendation', '?')} |"
            )
            if flags:
                print(f"  Row {row_label} 诊断：{flags}")


def _print_item_id_list(
    label: str,
    items: list[dict],
    reasons: dict[str, str],
) -> None:
    print(f"{label}（{len(items)}题）：")
    if not items:
        print("  - 无")
        return
    for item in items:
        item_id = str(item.get("item_id") or "?")
        reason = reasons.get(item_id) or "未记录原因"
        print(f"  - {item_id}：{reason}")


def print_selection_summary(proposed_update: dict) -> None:
    selection = proposed_update.get("selection_results") or {}
    reasons = proposed_update.get("selection_reasons") or {}
    selected = proposed_update.get("selected_items") or []
    reserve = proposed_update.get("reserve_items") or []
    repair_by_id = {
        str(entry.get("item_id")): entry
        for entry in [
            *(proposed_update.get("items_to_revise") or []),
            *(proposed_update.get("items_to_regenerate") or []),
        ]
        if isinstance(entry, dict) and entry.get("item_id")
    }
    repair_order = selection.get("revision_item_ids") or list(repair_by_id)
    revise = [
        repair_by_id[str(item_id)]
        for item_id in repair_order
        if str(item_id) in repair_by_id
    ]
    deferred = proposed_update.get("items_deferred_for_revision") or []
    removed_by_id = {
        str(item.get("item_id")): item
        for item in proposed_update.get("removed_items") or []
        if isinstance(item, dict) and item.get("item_id")
    }
    current_removed = [
        removed_by_id[item_id]
        for item_id in selection.get("removed_item_ids") or []
        if item_id in removed_by_id
    ]

    print("\n===== 心理测量返修诊断结果 =====")
    print(
        f"状态：{selection.get('status', '?')}；"
        f"证据等级：{selection.get('evidence_level', '?')}"
    )
    suppressed = selection.get("automatic_removal_suppressed") is True
    sample_size = selection.get("sample_size")
    minimum = selection.get("automatic_selection_minimum_sample_size")
    if suppressed:
        print(
            "为何没有自动删除："
            f"样本量 {sample_size} 低于自动筛选阈值 {minimum}，"
            "retain/revise/remove 仅作探索性报告。"
        )
    else:
        print(
            "自动删除与返修：已启用；"
            f"样本量 {sample_size} 达到阈值 {minimum}。"
        )
    locked_count = selection.get("locked_retained_count")
    if isinstance(locked_count, int):
        print(
            f"通过资格锁定：{locked_count}题；"
            "锁定版本继续参与整体统计与组卷，但不再自动返修。"
        )
    admission = selection.get("monotonic_admission")
    if isinstance(admission, dict):
        print(
            "组合准入："
            f"{admission.get('decision', '?')}；"
            f"{admission.get('reason', '未记录原因')}。"
        )
        admitted_metrics = admission.get("admitted_metrics")
        if isinstance(admitted_metrics, dict):
            print(
                "当前历史最佳正式组合："
                f"区分差值="
                f"{admitted_metrics.get('worst_case_discriminant_margin')}；"
                f"目标相关="
                f"{admitted_metrics.get('worst_case_target_rho')}；"
                f"Cronbach α="
                f"{admitted_metrics.get('worst_case_cronbach_alpha')}。"
            )
    developed_count = selection.get("developed_candidate_count")
    if isinstance(developed_count, int):
        print(
            f"累计开发候选：{developed_count} 题；"
            "按返修轮次与蓝图完成条件收敛，不设固定 25 题上限。"
        )

    _print_item_id_list("正式题", selected, reasons)
    _print_item_id_list("备用题", reserve, reasons)
    _print_item_id_list("返修题", revise, reasons)
    _print_item_id_list("待后续返修题（不阻塞组卷）", deferred, reasons)
    _print_item_id_list("本轮退出题", current_removed, reasons)
    if revise:
        print("返修意见（将自动传给出题专家）：")
        for entry in revise:
            review = entry.get("review")
            if not isinstance(review, dict):
                continue
            print(f"\n  题目 {entry.get('item_id', '?')}：")
            print_review_candidate(
                {
                    "current_item_review": review,
                    "current_item_repair_attempted": False,
                }
            )

    effects = selection.get("next_effect") or {}
    print("后续流程触发：")
    print(
        "  - 补题："
        + ("是" if effects.get("supplement_items") else "否")
    )
    print(
        "  - 返回出题专家修改："
        + ("是" if effects.get("repair_items") else "否")
    )
    print(
        "  - 重新模拟并分析："
        + ("是" if effects.get("reanalyze_after_bank_change") else "否")
    )
    print(
        "  - 再次组卷："
        + ("是" if effects.get("reassemble") else "否")
    )
    optimizer = (
        proposed_update.get("blueprint_coverage") or {}
    ).get("optimizer") or {}
    if optimizer:
        print(
            "组合选择："
            f"{optimizer.get('method')}；"
            f"候选组合={optimizer.get('combination_count', 0)}；"
            f"已执行组合优化={'是' if optimizer.get('optimized') else '否'}"
        )
        selected_metrics = optimizer.get("selected_metrics") or {}
        if selected_metrics:
            print(
                "正式组合指标："
                "最差模式目标相关="
                f"{_format_metric(selected_metrics.get('worst_case_target_rho'))}；"
                "最差模式Cronbach α="
                f"{_format_metric(selected_metrics.get('worst_case_cronbach_alpha'))}；"
                "区分差值="
                f"{_format_metric(selected_metrics.get('worst_case_discriminant_margin'))}"
            )


def print_workflow_effect(action: str, proposed_update: dict) -> None:
    if action == "simulate_responses":
        summary = proposed_update.get("virtual_response_summary") or {}
        print(
            "\n[虚拟作答] "
            f"复用未变题作答 {summary.get('reused_sjt_records', 0)} 条；"
            f"新增SJT调用 {summary.get('scheduled_sjt_api_calls', 0)} 次；"
            f"复用人格总结 "
            f"{summary.get('reused_persona_summary_records', 0)} 条；"
            f"复用Neo-FFI作答 "
            f"{summary.get('reused_neo_ffi_records', 0)} 条。"
        )
    elif action == "assemble_test":
        assembled = proposed_update.get("assembled_test") or {}
        round_number = proposed_update.get("reassembly_round", 0)
        print(
            "\n[流程触发] 已完成"
            + ("再次组卷" if round_number else "首次组卷")
            + f"：正式题 {assembled.get('item_count', '?')} 道。"
        )


def print_user_progress_event(
    event: TraceEvent,
    update: dict,
    state: dict,
    previous_state: dict | None = None,
) -> None:
    """Render stable, user-facing stage and item lifecycle progress."""

    node = event.get("node")
    action = str(event.get("action") or "unknown")
    event_type = event.get("event_type")
    if node in ACTION_START_NODES and event_type == "completed":
        if action == "finish":
            print("\n===== 工作流全部完成 =====", flush=True)
        else:
            print(
                f"\n===== 开始：{item_stage_label(action, state)} =====",
                flush=True,
            )
        return

    if node == "execute":
        if event_type == "completed":
            duration_ms = event.get("duration_ms")
            duration = (
                f"，耗时 {duration_ms / 1000:.1f} 秒"
                if isinstance(duration_ms, (int, float))
                else ""
            )
            print(
                f"\n===== 完成：{item_stage_label(action, state)}{duration} =====",
                flush=True,
            )
            print_automatic_item_result(action, update, state)
            proposed_update = update.get("pending_state_update") or {}
            if action == "analyze_psychometrics":
                print_psychometric_summary(proposed_update, state)
            elif action == "select_items":
                print_selection_summary(proposed_update)
            print_workflow_effect(action, proposed_update)
        elif event_type == "failed":
            print(
                f"\n===== 失败：{item_stage_label(action, state)} =====\n"
                f"{event.get('error', '未知错误')}",
                flush=True,
            )
        return

    if (
        node == "accept_item"
        and event_type == "completed"
        and state.get("item_development_mode") == "automatic"
    ):
        print("\n===== 题目通过并进入候选题库 =====")
        print_item_candidate(
            (previous_state or {}).get("current_item")
            or state.get("current_item")
        )
    elif node == "abandon_item" and event_type == "completed":
        was_psychometric_candidate = isinstance(
            (previous_state or {}).get("active_psychometric_repair"), dict
        )
        current_item = (
            (previous_state or {}).get("current_item")
            or state.get("current_item")
            or {}
        )
        heading = (
            "返修候选未通过，已保留上一有效版本"
            if was_psychometric_candidate
            else "题目候选淘汰"
        )
        print(
            f"\n===== {heading} =====\n"
            f"题目编号：{current_item.get('item_id', '未知')}",
            flush=True,
        )


def print_heartbeat(action: object, elapsed_seconds: float) -> None:
    print(
        f"\n[运行中] {action_label(action)}仍在执行，"
        f"已等待 {round(elapsed_seconds)} 秒……",
        flush=True,
    )


def print_trace_event(event: TraceEvent) -> None:
    """以便于扫描的格式输出一个新的轨迹事件。"""

    step = event.get("step", "?")
    node = event.get("node", "unknown")
    action = event.get("action", "unknown")
    event_type = event.get("event_type", "unknown")
    duration_ms = event.get("duration_ms")
    duration = f" ({duration_ms} ms)" if duration_ms is not None else ""

    print(f"\n[{step}] {node} → {action}: {event_type}{duration}")
    if event.get("reason"):
        print(f"    原因：{event['reason']}")
    if event.get("error"):
        print(f"    错误：{event['error']}")

    state_changes = event.get("state_changes", {})
    if state_changes:
        print("    State changes:")
        for field, change in state_changes.items():
            print(f"      - {field}")
            print("        before:")
            pprint(change.get("before"), indent=10, width=100)
            print("        after:")
            pprint(change.get("after"), indent=10, width=100)


def print_test_specification(specification: object) -> None:
    """以用户可读形式展示测验规格，不暴露 State 结构。"""

    if not isinstance(specification, dict):
        print("暂未形成测验规格。")
        return
    for field, label in SPECIFICATION_LABELS.items():
        value = specification.get(field)
        if isinstance(value, list):
            value = "；".join(value) if value else "无"
        print(f"  {label}：{value}")


def print_blueprint_candidate(blueprint: object) -> None:
    if not isinstance(blueprint, dict) or blueprint.get("version") != 7:
        return
    retained = planned_retention_count(blueprint)
    generated = planned_generation_count(blueprint)
    print(
        "构念—题目细目表："
        f"最终保留 {retained} 题，"
        f"计划生成 {generated} 题"
    )
    profile = blueprint.get("construct_profile_snapshot") or {}
    print(
        f"构念档案：{profile.get('inventory_name', '?')} / "
        f"{profile.get('domain_name', '?')} / "
        f"{profile.get('selection_level', '?')}"
    )
    for facet in profile.get("facets") or []:
        if isinstance(facet, dict):
            print(
                f"  - {facet.get('facet_name', facet.get('facet_id', '?'))}："
                f"{facet.get('definition', '')}"
            )
    print("维度单元：")
    for cell in blueprint.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        print(
            f"  - {cell.get('facet_id', '?')} / "
            f"{cell.get('behavior_id', '?')} / "
            f"{cell.get('mechanism_id', '?')} / "
            f"{cell.get('situation_id', '?')} / "
            + f"生成 {cell.get('planned_generation_count', '?')}，"
            f"保留 {cell.get('planned_retention_count', '?')}"
        )
    print("固定题目槽位：")
    for slot in blueprint.get("slots") or []:
        if isinstance(slot, dict):
            print(
                f"  - {slot.get('specification_id', '?')} / "
                f"{slot.get('blueprint_cell_id', '?')}"
            )


def print_skeleton_candidate(update: dict, state: dict) -> None:
    """Render the committed abstract skeleton before its concrete item."""

    current = update.get("current_item_specification") or {}
    specification_id = current.get("specification_id")
    skeletons = update.get("item_skeletons") or {}
    skeleton = skeletons.get(specification_id)
    if not isinstance(skeleton, dict):
        return
    blueprint = state.get("blueprint") or {}
    profile = blueprint.get("construct_profile_snapshot") or {}
    facet = next(
        (
            row for row in profile.get("facets") or []
            if isinstance(row, dict)
            and row.get("facet_id") == current.get("target_dimension_id")
        ),
        {},
    )
    print("\n===== 当前心理骨架 =====")
    print(f"固定槽位：{specification_id or '未知'}")
    print(f"目标 facet：{facet.get('facet_name', current.get('target_dimension_id', '?'))}")
    print(f"行为证据：{current.get('behavior_evidence_id', '?')}")
    print(f"激活机制：{current.get('activation_mechanism', '?')}")
    print(f"情境引用：{current.get('situation_id', '?')}")
    print(f"情境类型：{skeleton.get('situation_type', '')}")
    print(f"风险水平：{skeleton.get('stakes_level', '')}")
    print(f"社会情境：{skeleton.get('social_context', '')}")
    print(f"核心冲突：{skeleton.get('behavioral_tension', '')}")
    print("四级行为结构：")
    for row in skeleton.get("option_structure") or []:
        if not isinstance(row, dict):
            continue
        print(
            f"  - {row.get('behavioral_level', '?')}："
            f"{row.get('behavioral_tendency', '')}；"
            f"心理功能：{row.get('psychological_function', '')}"
        )
    print("骨架校验：程序确定性校验通过；未执行独立 LLM 骨架审核")


def print_item_candidate(item: object) -> None:
    if not isinstance(item, dict):
        return
    print(f"题目编号：{item.get('item_id', '未编号')}")
    print(f"情境：{item.get('scenario', '')}")
    print(f"问题：{item.get('response_instruction', '')}")
    scoring_key = item.get("scoring_key") or {}
    print("选项：")
    for option in item.get("response_options") or []:
        if not isinstance(option, dict):
            continue
        option_id = option.get("option_id", "?")
        score = scoring_key.get(option_id, "?")
        print(f"  {option_id}. {option.get('text', '')}（计分：{score}）")


def print_review_candidate(update: dict) -> None:
    review = update.get("current_item_review")
    if isinstance(review, dict):
        tasks = review.get("repair_tasks") or []
        decision = derive_item_review_decision(
            review,
            repair_attempted=bool(
                update.get("current_item_repair_attempted")
            ),
        )
        print(f"审题结论：{decision}")
        summary = review.get("summary")
        if summary:
            print(f"审题摘要：{summary}")
        findings = review.get("findings") or []
        if findings:
            print("四维诊断：")
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            option_ids = finding.get("affected_option_ids") or []
            locus = str(finding.get("locus") or "?")
            location = (
                f"{locus}[{','.join(option_ids)}]"
                if option_ids
                else locus
            )
            print(
                f"  - [{finding.get('severity', '?')}] "
                f"{finding.get('criterion', '?')} / {location}："
                f"{finding.get('problem', '')}"
            )
            evidence = finding.get("evidence")
            if evidence:
                print(f"    依据：{evidence}")
        if tasks:
            print("程序派生修复任务：")
        for task in tasks:
            if not isinstance(task, dict):
                continue
            target_labels = []
            for target in task.get("targets") or []:
                if not isinstance(target, dict):
                    continue
                field = target.get("field", "?")
                option_ids = target.get("option_ids") or []
                target_labels.append(
                    f"{field}[{','.join(option_ids)}]"
                    if option_ids
                    else str(field)
                )
            print(
                f"  - 修复任务 {task.get('task_id', '?')}："
                f"{', '.join(target_labels)}"
            )
            print(f"    问题：{task.get('problem', '')}")
            print(f"    修改要求：{task.get('instruction', '')}")
        return

    print("审题结果无效：缺少 current_item_review")


def print_candidate(payload: dict) -> None:
    """按业务动作展示用户需要审核的内容。"""

    action = payload.get("action")
    update = payload.get("proposed_update") or {}
    if action == "build_blueprint":
        print_blueprint_candidate(update.get("blueprint"))
    elif action in {"generate_item", "revise_item", "regenerate_item"}:
        print_skeleton_candidate(update, payload)
        print_item_candidate(update.get("current_item"))
    elif action == "review_item":
        print_review_candidate(update)


def prompt_requirement_decision(payload: dict) -> dict:
    """展示需求缺口、问题和建议，并收集一轮自然语言答复。"""

    proposed_update = payload.get("proposed_update") or {}
    print("\n===== 测验需求澄清 =====")
    if payload.get("summary"):
        print(f"摘要：{payload['summary']}")
    if payload.get("validation_error"):
        print(f"上一次输入无效：{payload['validation_error']}")
    if payload.get("field_errors"):
        print("字段问题：")
        for field, error in payload["field_errors"].items():
            label = SPECIFICATION_LABELS.get(field, field)
            print(f"  - {label}: {error}")

    print("\n当前候选测验规格：")
    candidate_specification = proposed_update.get("test_specification")
    print_test_specification(candidate_specification)
    if isinstance(candidate_specification, dict):
        try:
            resolved_profile = resolve_specification_profile(
                candidate_specification
            )
        except ValueError:
            resolved_profile = None
        if resolved_profile is not None:
            print(
                "  构念库解析："
                f"{resolved_profile['inventory_name']} / "
                f"{resolved_profile['domain_name']} / "
                f"{resolved_profile['selection_level']}，"
                f"{len(resolved_profile['facets'])} 个 facet"
            )
    questions = payload.get("questions") or []
    suggestions = payload.get("suggestions") or []
    if questions:
        print("\n请补充以下信息：")
        for index, question in enumerate(questions, 1):
            text = question.get("text") if isinstance(question, dict) else question
            print(f"  {index}. {text}")
    if suggestions:
        print("\n系统建议：")
        for suggestion in suggestions:
            print(f"  - {suggestion.get('field')}")
            print(f"    理由：{suggestion.get('reason')}")

    decisions = payload.get("available_decisions") or []
    if "confirm" in decisions:
        print("\n需求字段已经完整，请选择：")
        print("  1. 确认规格并进入生成计划")
        print("  2. 用自然语言提出修改")
        print("  3. 停止工作流")
        while True:
            choice = input("你的选择 [1-3]：").strip()
            if choice == "1":
                return {"decision": "confirm", "feedback": None}
            if choice == "2":
                feedback = input("请说明需要修改的内容：").strip()
                if feedback:
                    return {"decision": "revise", "feedback": feedback}
                print("修改需求时必须提供具体内容")
                continue
            if choice == "3":
                feedback = input("停止原因（可留空）：").strip() or None
                return {"decision": "stop", "feedback": feedback}
            print("请输入 1、2 或 3")

    print("\n请选择：")
    print("  1. 回答上述问题或补充需求")
    if "accept_suggestions" in decisions:
        print("  2. 接受本轮全部系统建议和默认推断值")
    print("  3. 停止工作流")
    while True:
        choice = input("你的选择 [1-3]：").strip()
        if choice == "1":
            feedback = input("请用自然语言回答：").strip()
            if feedback:
                return {"decision": "answer", "feedback": feedback}
            print("回答不能为空")
            continue
        if choice == "2" and "accept_suggestions" in decisions:
            return {"decision": "accept_suggestions", "feedback": None}
        if choice == "3":
            feedback = input("停止原因（可留空）：").strip() or None
            return {"decision": "stop", "feedback": feedback}
        print("请输入有效选项")


def prompt_virtual_sample_selection(payload: dict) -> dict:
    """展示内置人格档案池，并让用户选择本次使用人数。"""

    pool = payload.get("pool") or {}
    recommendations = payload.get("recommendations") or []
    recommended_size = payload.get("recommended_sample_size")
    default_seed = payload.get("default_seed", 7)
    default_max_concurrency = payload.get("default_max_concurrency", 5)
    default_max_retries = payload.get("default_max_retries", 2)

    print("\n===== 选择虚拟被试人数 =====")
    print(
        f"可用匿名人格档案：{pool.get('available_count', '?')} 名；"
        f"每人 {pool.get('item_count', '?')} 道人格作答。"
    )
    facet_counts = pool.get("facet_item_counts") or {}
    if facet_counts:
        print(
            "人格题构成："
            + "、".join(
                f"{facet} {count}题"
                for facet, count in facet_counts.items()
            )
        )
    if payload.get("method_note"):
        print(f"说明：{payload['method_note']}")
    print(
        f"模型请求控制：最大并发 {default_max_concurrency}，"
        f"失败后最多重试 {default_max_retries} 次。"
    )
    if payload.get("validation_error"):
        print(f"上一次输入无效：{payload['validation_error']}")

    recommended_index = None
    print("\n推荐档位：")
    for index, option in enumerate(recommendations, 1):
        marker = "（推荐）" if option.get("recommended") else ""
        print(
            f"  {index}. {option.get('label')}："
            f"{option.get('sample_size')} 名{marker}"
        )
        print(f"     {option.get('description')}")
        if option.get("recommended"):
            recommended_index = index
    custom_index = len(recommendations) + 1
    print(f"  {custom_index}. 自定义人数")

    prompt_suffix = (
        f"，直接回车使用推荐值 {recommended_size}"
        if recommended_size is not None
        else ""
    )
    while True:
        choice = input(
            f"你的选择 [1-{custom_index}{prompt_suffix}]："
        ).strip()
        if not choice and recommended_size is not None:
            return {
                "sample_size": recommended_size,
                "seed": default_seed,
                "max_concurrency": default_max_concurrency,
                "max_retries": default_max_retries,
            }
        if choice.isdigit():
            selected_index = int(choice)
            if 1 <= selected_index <= len(recommendations):
                return {
                    "sample_size": recommendations[selected_index - 1][
                        "sample_size"
                    ],
                    "seed": default_seed,
                    "max_concurrency": default_max_concurrency,
                    "max_retries": default_max_retries,
                }
            if selected_index == custom_index:
                custom_value = input(
                    "请输入希望使用的虚拟被试人数："
                ).strip()
                if custom_value.isdigit():
                    return {
                        "sample_size": int(custom_value),
                        "seed": default_seed,
                        "max_concurrency": default_max_concurrency,
                        "max_retries": default_max_retries,
                    }
                print("人数必须是正整数")
                continue
        default_hint = (
            f"；推荐档位是 {recommended_index}"
            if recommended_index is not None
            else ""
        )
        print(f"请输入 1 到 {custom_index}{default_hint}")


def prompt_user_decision(payload: dict) -> dict:
    """按任务类型收集一次有效的用户决策。"""

    if payload.get("type") == "virtual_sample_selection":
        return prompt_virtual_sample_selection(payload)

    if payload.get("type") == "post_virtual_response_decision":
        print("\n===== 虚拟被试作答完成 =====")
        print(payload.get("summary") or "")
        print("  1. 开始根据心理测量结果修改题目")
        print("  2. 暂不修改，继续后续流程")
        print("  3. 停止本次运行")
        while True:
            choice = input("你的选择 [1-3]：").strip()
            if choice == "1":
                return {"decision": "start"}
            if choice == "2":
                return {"decision": "skip"}
            if choice == "3":
                return {"decision": "stop"}
            print("请输入 1、2 或 3")

    if payload.get("type") == "psychometric_repair_confirmation":
        print("\n===== 心理测量返修确认 =====")
        print(f"题目编号：{payload.get('item_id', '?')}")
        item = payload.get("item")
        if isinstance(item, dict):
            print_item_candidate(item)
        diagnosis = payload.get("diagnosis") or {}
        if diagnosis.get("summary"):
            print(f"诊断摘要：{diagnosis['summary']}")
        candidates = diagnosis.get("candidate_diagnoses") or []
        if candidates:
            print("候选问题：")
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                options = ",".join(candidate.get("affected_option_ids") or [])
                location = candidate.get("suspect_components") or []
                print(
                    f"  - {candidate.get('diagnosis_id', '?')} / "
                    f"{','.join(location)}[{options}] / "
                    f"置信度={candidate.get('confidence', '?')}"
                )
                print(f"    证据：{candidate.get('textual_evidence', '')}")
                print(f"    说明：{candidate.get('explanation', '')}")
        tasks = diagnosis.get("repair_tasks") or []
        if tasks:
            print("已确认的修改任务：")
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                edit = task.get("atomic_edit") or {}
                option_ids = ",".join(edit.get("option_ids") or [])
                print(
                    f"  - {task.get('diagnosis_id', '?')} / "
                    f"{edit.get('target_field', '?')}[{option_ids}]："
                    f"{edit.get('problem', '')}"
                )
                print(f"    修改要求：{edit.get('instruction', '')}")
        print("  1. 确认全部任务，逐条修改后统一重测")
        print("  2. 跳过本题修改并保留警告")
        print("  3. 停止本次运行")
        while True:
            choice = input("你的选择 [1-3]：").strip()
            if choice == "1":
                return {"decision": "approve"}
            if choice == "2":
                return {"decision": "skip"}
            if choice == "3":
                return {"decision": "stop"}
            print("请输入 1、2 或 3")

    if payload.get("type") == "item_development_mode_selection":
        print("\n===== 选择题目开发模式 =====")
        for index, mode in enumerate(payload.get("modes") or [], 1):
            print(f"  {index}. {mode.get('label')}")
            print(f"     {mode.get('description')}")
        while True:
            choice = input("你的选择 [1-2]：").strip()
            if choice == "1":
                return {"mode": "manual"}
            if choice == "2":
                return {"mode": "automatic"}
            print("请输入 1 或 2")

    if payload.get("type") == "requirement_confirmation":
        return prompt_requirement_decision(payload)

    print("\n===== 请审核候选结果 =====")
    if payload.get("summary"):
        print(f"摘要：{payload['summary']}")
    if payload.get("validation_error"):
        print(f"上一次输入无效：{payload['validation_error']}")

    print_candidate(payload)
    print("\n请选择：")
    print("  1. 通过并继续")
    print("  2. 提供意见并重新生成")
    print("  3. 停止工作流")

    while True:
        choice = input("你的选择 [1-3]：").strip()
        if choice == "1":
            return {
                "decision": "approve",
                "feedback": None,
                "state_patch": None,
            }
        if choice == "2":
            feedback = input("请说明需要如何调整：").strip()
            if not feedback:
                print("重新生成时必须提供调整意见")
                continue
            return {
                "decision": "regenerate",
                "feedback": feedback,
                "state_patch": None,
            }
        if choice == "3":
            feedback = input("停止原因（可留空）：").strip() or None
            return {
                "decision": "stop",
                "feedback": feedback,
                "state_patch": None,
            }
        print("请输入 1、2 或 3")


def get_interrupt_payload(interrupt_update: object) -> dict:
    """从 LangGraph 的中断更新中提取用户可见负载。"""

    interrupts = (
        list(interrupt_update)
        if isinstance(interrupt_update, (list, tuple))
        else [interrupt_update]
    )
    if not interrupts:
        raise ValueError("LangGraph 返回了空的 interrupt 更新")

    payload = getattr(interrupts[0], "value", interrupts[0])
    if not isinstance(payload, dict):
        raise ValueError("LangGraph interrupt 负载必须是对象")
    return payload


async def run_with_trace(
    initial_state: dict,
    *,
    debug: bool = False,
    heartbeat_interval_seconds: float | None = None,
    checkpoint_root: Path | None = None,
) -> dict:
    """流式执行图，并在每个 Agent 结果后暂停等待用户确认。"""

    if heartbeat_interval_seconds is None:
        heartbeat_interval_seconds = float(
            os.getenv("SJT_HEARTBEAT_INTERVAL_SECONDS", "20")
        )
    if heartbeat_interval_seconds <= 0:
        raise ValueError("heartbeat_interval_seconds 必须是正数")

    result = dict(initial_state)
    displayed_event_ids: set[str] = set()
    config = {"configurable": {"thread_id": initial_state["run_id"]}}
    graph_input: dict | Command = initial_state

    with progress_callback(print_runtime_progress):
        while True:
            resume_command: Command | None = None
            stream = app.astream(
                graph_input,
                config=config,
                stream_mode="updates",
            )
            iterator = stream.__aiter__()
            next_chunk_task: asyncio.Task | None = None
            wait_started_at = monotonic()
            try:
                while True:
                    if next_chunk_task is None:
                        next_chunk_task = asyncio.create_task(anext(iterator))
                    done, _ = await asyncio.wait(
                        {next_chunk_task},
                        timeout=heartbeat_interval_seconds,
                    )
                    if not done:
                        active_action = (
                            result.get("pending_action")
                            or (result.get("route") or {}).get("next_action")
                            or "初始化工作流"
                        )
                        print_heartbeat(
                            active_action,
                            monotonic() - wait_started_at,
                        )
                        continue

                    try:
                        chunk = next_chunk_task.result()
                    except StopAsyncIteration:
                        break
                    finally:
                        next_chunk_task = None
                    wait_started_at = monotonic()

                    if "__interrupt__" in chunk:
                        payload = get_interrupt_payload(
                            chunk["__interrupt__"]
                        )
                        resume_command = Command(
                            resume=prompt_user_decision(payload)
                        )
                        break

                    for update in chunk.values():
                        if not isinstance(update, dict):
                            continue
                        previous_result = dict(result)
                        result.update(update)
                        if checkpoint_root is not None:
                            save_run_checkpoint(
                                result,
                                checkpoint_root=Path(checkpoint_root),
                            )
                        for event in update.get("execution_history", []):
                            event_id = event.get("event_id")
                            if (
                                event_id
                                and event_id in displayed_event_ids
                            ):
                                continue
                            print_user_progress_event(
                                event,
                                update,
                                result,
                                previous_result,
                            )
                            if debug:
                                print_trace_event(event)
                            if event_id:
                                displayed_event_ids.add(event_id)
            finally:
                if next_chunk_task is not None:
                    next_chunk_task.cancel()
                    await asyncio.gather(
                        next_chunk_task,
                        return_exceptions=True,
                    )
                aclose = getattr(stream, "aclose", None)
                if callable(aclose):
                    await aclose()

            if resume_command is None:
                break
            graph_input = resume_command

    return result


def print_final_result(result: dict) -> None:
    """只展示流程状态和可交付结果，不打印内部 State。"""

    status = result.get("status", "unknown")
    print("\n===== 本次运行结束 =====")
    if status == "failed":
        errors = result.get("errors") or []
        message = errors[-1].get("message") if errors else "未知错误"
        print(f"运行失败：{message}")
        return
    if status == "stopped":
        print("工作流已停止。")
        return
    selection = result.get("selection_results") or {}
    provisional_count = len(selection.get("provisional_item_ids") or [])
    selected_count = len(result.get("selected_items") or [])
    reserve_count = len(result.get("reserve_items") or [])
    if result.get("final_test"):
        print(
            "开发版测验已经生成。"
            if selection.get("developmental_override")
            else "候选测验已经生成。"
        )
    else:
        print(f"运行状态：{status}")
    print(f"开发题库：{len(result.get('item_pool') or [])} 题")
    if selected_count or reserve_count:
        print(
            f"入卷：{selected_count} 题；备用：{reserve_count} 题；"
            f"开发版标记：{provisional_count} 题"
        )
    if result.get("item_bank_id"):
        print(
            "冻结题库："
            f"{result['item_bank_id']}（v{result.get('item_bank_version')}）"
        )
    deliverables = [
        (
            "正式测验",
            (result.get("final_test") or {}).get("file_path"),
        ),
        (
            "技术报告",
            (result.get("technical_report") or {}).get("markdown_path"),
        ),
        ("题库文件", result.get("item_database_ref")),
        (
            "虚拟被试报告",
            (result.get("virtual_respondent_report") or {}).get(
                "file_path"
            ),
        ),
    ]
    available = [
        (label, path)
        for label, path in deliverables
        if isinstance(path, str) and path
    ]
    if available:
        print("交付文件：")
        for label, path in available:
            print(f"  - {label}：{path}")


def select_start_state(
    new_state_factory: Callable[[], dict],
    *,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict:
    """Select a fresh run or resume the newest nonterminal checkpoint."""

    latest = find_latest_resumable_checkpoint(checkpoint_root)
    if latest is None:
        return new_state_factory()
    state = latest["state"]
    errors = state.get("errors") or []
    latest_error = (
        errors[-1].get("message")
        if isinstance(errors[-1], dict)
        else str(errors[-1])
    ) if errors else "无"
    print(
        "\n===== 检测到未完成运行 =====\n"
        f"运行编号：{latest['run_id']}\n"
        f"保存时间：{latest['saved_at']}\n"
        f"当前阶段：{state.get('current_phase', 'unknown')}\n"
        f"候选题目：{len(state.get('item_pool') or [])}\n"
        f"最近错误：{latest_error}\n"
        "  1. 继续上次运行\n"
        "  2. 放弃上次运行并开始新运行",
        flush=True,
    )
    while True:
        choice = input("你的选择 [1-2]：").strip()
        if choice == "1":
            return prepare_resumed_state(state)
        if choice == "2":
            abandoned = dict(state)
            abandoned["status"] = "stopped"
            save_run_checkpoint(
                abandoned,
                checkpoint_root=Path(checkpoint_root),
            )
            return new_state_factory()
        print("请输入 1 或 2")


async def main() -> None:
    global app
    state = select_start_state(
        lambda: create_initial_state(
            "请帮我出一个测量gregariousness的人格情境判断测验，测量大学生的，用于人格测评，一共16道题",
            target_population=None,
            target_construct=None,
            requested_item_count=None,
            # 为单题生成、审查和可能的修改循环保留足够执行步数。
            max_steps=1000,
        )
    )
    debug = os.getenv("SJT_DEBUG", "").strip().lower() in {"1", "true", "yes"}
    while True:
        result = await run_with_trace(
            state,
            debug=debug,
            checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
        )
        print_final_result(result)
        if result.get("status") != "failed":
            return
        while True:
            choice = input(
                "运行已暂停。输入 1 从最近检查点重试，"
                "输入 2 停止本次运行 [1-2]："
            ).strip()
            if choice == "1":
                state = prepare_retry_state(
                    result,
                    checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
                )
                app = build_sjt_graph()
                break
            if choice == "2":
                stopped = dict(result)
                stopped["status"] = "stopped"
                save_run_checkpoint(
                    stopped,
                    checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
                )
                return
            print("请输入 1 或 2")

if __name__ == "__main__":
    asyncio.run(main())
