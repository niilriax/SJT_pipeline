"""Streamlit workspace for developing an SJT through the full workflow."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import streamlit as st
from langgraph.types import Command

from sjt_system.authoring.construct_registry import (
    resolve_specification_profile,
)
from sjt_system.authoring.generation_plan import (
    planned_generation_count,
    planned_retention_count,
)
from sjt_system.runtime.checkpoint import (
    DEFAULT_CHECKPOINT_ROOT,
    find_latest_resumable_checkpoint,
    prepare_retry_state,
    prepare_resumed_state,
)
from sjt_system.state import create_initial_state
from sjt_system.ui.presenters import (
    action_label,
    build_timeline_entries,
    extract_action_content,
    phase_label,
    progress_summary,
)
from sjt_system.ui.workflow_runner import WorkflowRunError, run_until_pause
from sjt_system.workflow.graph import build_sjt_graph


SESSION_DEFAULTS = {
    "sjt_state": None,
    "sjt_interrupt": None,
    "sjt_timeline": [],
    "sjt_seen_event_ids": set(),
    "sjt_runtime_events": [],
    "sjt_error": None,
    "sjt_latest_checkpoint": None,
    "sjt_workflow": None,
}

SPECIFICATION_LABELS = {
    "construct_selection": "测量构念",
    "target_population": "目标人群",
    "final_item_count": "最终题量",
    "output_language": "输出语言",
}

CONTENT_LABELS = {
    "virtual_response_summary": "虚拟作答摘要",
    "test_statistics": "测验统计",
    "item_statistics": "题目统计",
    "factor_results": "因素分析",
    "irt_results": "IRT 分析",
    "dif_results": "DIF 分析",
    "selection_results": "筛选结果",
    "selected_items": "入选题目",
    "assembled_test": "组卷结果",
    "final_test": "正式测验",
    "technical_report": "技术报告",
    "virtual_respondent_report": "虚拟被试报告",
}

USER_VISIBLE_TIMELINE_NODES = {
    "execute",
    "accept_item",
    "abandon_item",
    "item_development_mode_selection",
    "virtual_sample_selection",
    "stop",
}


def _workflow():
    workflow = st.session_state.sjt_workflow
    if workflow is None:
        workflow = build_sjt_graph()
        st.session_state.sjt_workflow = workflow
    return workflow


def _initialize_session() -> None:
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = (
                set(default) if isinstance(default, set)
                else list(default) if isinstance(default, list)
                else default
            )


def _reset_session() -> None:
    for key, default in SESSION_DEFAULTS.items():
        st.session_state[key] = (
            set(default) if isinstance(default, set)
            else list(default) if isinstance(default, list)
            else default
        )


def _apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 85% 0%, #eef6f3 0, transparent 30rem),
                #f7f8f6;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 5rem;
        }
        h1, h2, h3 { letter-spacing: -0.025em; }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid #e2e7e3;
            border-radius: 14px;
            padding: 0.85rem 1rem;
        }
        [data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid #dde5df;
            border-radius: 18px;
            padding: 1.2rem 1.35rem 0.5rem;
            box-shadow: 0 12px 36px rgba(27, 54, 43, 0.06);
        }
        [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.82);
            border-color: #e0e5e1;
            border-radius: 14px;
        }
        .sjt-kicker {
            color: #28705a;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }
        .sjt-subtitle {
            color: #5c6862;
            font-size: 1.05rem;
            max-width: 760px;
            margin-top: -0.4rem;
            margin-bottom: 1.8rem;
        }
        .sjt-note {
            color: #617068;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _runtime_message(event: Mapping[str, Any]) -> str:
    event_type = event.get("type")
    if event_type == "simulation_progress":
        return (
            f"虚拟作答：{event.get('completed', 0)}/"
            f"{event.get('total', '?')}"
        )
    if event_type == "simulation_stage":
        return (
            f"虚拟作答 · {event.get('stage', '处理中')} · "
            f"{event.get('status', 'running')}"
        )
    if event_type in {"request_retry", "output_repair"}:
        return (
            f"{event.get('job_label', '模型请求')}正在重试："
            f"{event.get('reason', '输出未通过校验')}"
        )
    if event_type == "action_fallback":
        return (
            f"{action_label(event.get('from_action'))}未通过校验，"
            f"转为{action_label(event.get('to_action'))}"
        )
    return str(event.get("message") or event_type or "正在处理")


def _latest_update_message(update: Mapping[str, Any]) -> str:
    history = update.get("execution_history") or []
    if history and isinstance(history[-1], Mapping):
        event = history[-1]
        return (
            f"{action_label(event.get('action'))} · "
            f"{event.get('event_type', 'completed')}"
        )
    return "流程状态已更新"


def _advance(graph_input: dict[str, Any] | Command) -> None:
    state = st.session_state.sjt_state
    if not isinstance(state, Mapping):
        raise ValueError("页面中没有可运行的工作流状态")

    with st.status("正在推进测验开发流程", expanded=True) as status:
        try:
            turn = asyncio.run(
                run_until_pause(
                    _workflow(),
                    graph_input,
                    state,
                    checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
                    on_update=lambda _node, update, _state: status.write(
                        _latest_update_message(update)
                    ),
                    on_progress=lambda event: status.write(
                        _runtime_message(event)
                    ),
                )
            )
            _store_turn_result(
                turn.state,
                turn.interrupt,
                turn.updates,
                turn.runtime_events,
            )
            if turn.state.get("status") == "failed":
                st.session_state.sjt_error = _latest_state_error(turn.state)
                status.update(
                    label="流程已暂停，可以从检查点重试",
                    state="error",
                    expanded=True,
                )
            else:
                st.session_state.sjt_error = None
                status.update(
                    label=(
                        "流程等待你的确认"
                        if turn.interrupt
                        else "本轮流程已经完成"
                    ),
                    state="complete",
                    expanded=False,
                )
        except WorkflowRunError as exc:
            _store_turn_result(
                exc.state,
                None,
                exc.updates,
                exc.runtime_events,
            )
            st.session_state.sjt_error = str(exc)
            status.update(
                label="流程运行失败",
                state="error",
                expanded=True,
            )
            st.exception(exc.original_error)
        except Exception as exc:
            st.session_state.sjt_error = str(exc)
            status.update(
                label="流程运行失败",
                state="error",
                expanded=True,
            )
            st.exception(exc)


def _store_turn_result(
    state: Mapping[str, Any],
    interrupt: Mapping[str, Any] | None,
    updates: list[dict[str, Any]],
    runtime_events: list[dict[str, Any]],
) -> None:
    entries, seen = build_timeline_entries(
        updates,
        st.session_state.sjt_seen_event_ids,
    )
    st.session_state.sjt_state = dict(state)
    st.session_state.sjt_interrupt = (
        dict(interrupt) if interrupt is not None else None
    )
    st.session_state.sjt_timeline.extend(entries)
    st.session_state.sjt_seen_event_ids = seen
    st.session_state.sjt_runtime_events.extend(runtime_events)


def _latest_state_error(state: Mapping[str, Any]) -> str:
    errors = state.get("errors") or []
    if not errors:
        return "流程未完成，请从最近检查点重试。"
    latest = errors[-1]
    if isinstance(latest, Mapping):
        return str(latest.get("message") or "流程未完成")
    return str(latest)


def _retry_current_run(state: Mapping[str, Any]) -> None:
    retry_state = prepare_retry_state(
        state,
        checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
    )
    st.session_state.sjt_state = retry_state
    st.session_state.sjt_interrupt = None
    st.session_state.sjt_error = None
    # Drop the in-memory graph checkpoint before replaying the durable state.
    st.session_state.sjt_workflow = build_sjt_graph()
    _advance(retry_state)


def _start_new_run(user_request: str) -> None:
    _reset_session()
    state = create_initial_state(user_request, max_steps=1000)
    st.session_state.sjt_state = state
    _advance(state)


def _resume_latest_run() -> None:
    latest = find_latest_resumable_checkpoint(DEFAULT_CHECKPOINT_ROOT)
    if latest is None:
        st.warning("没有找到可继续的运行。")
        return
    _reset_session()
    state = prepare_resumed_state(latest["state"])
    st.session_state.sjt_state = state
    _advance(state)


def _render_start_page() -> None:
    left, right = st.columns([1.55, 0.7], gap="large")
    with left:
        st.markdown(
            '<div class="sjt-kicker">SJT DEVELOPMENT WORKSPACE</div>',
            unsafe_allow_html=True,
        )
        st.title("把测验需求变成可审查的完整成果")
        st.markdown(
            '<div class="sjt-subtitle">'
            "描述你要测量什么、面向谁以及希望得到多少道题。"
            "系统会逐步完成需求澄清、构念档案解析、生成计划、题目开发、"
            "虚拟检验和报告生成。"
            "</div>",
            unsafe_allow_html=True,
        )
        with st.form("new_sjt_request"):
            request = st.text_area(
                "你想开发什么测验？",
                height=170,
                placeholder=(
                    "例如：开发一套面向高校辅导员的危机沟通情境判断"
                    "测验，用于培训评估，共 12 道题……"
                ),
                label_visibility="visible",
            )
            st.markdown(
                '<div class="sjt-note">'
                "暂时不确定的内容可以留给系统在下一步与你确认。"
                "</div>",
                unsafe_allow_html=True,
            )
            submitted = st.form_submit_button(
                "开始开发",
                type="primary",
                use_container_width=True,
            )
        if submitted:
            if not request.strip():
                st.error("请先描述你的测验需求。")
            else:
                _start_new_run(request.strip())
                st.rerun()

    with right:
        st.subheader("你会看到什么")
        st.markdown(
            "1. **当前进度**：流程进行到了哪一步\n\n"
            "2. **阶段成果**：模型、蓝图、题目和分析结果\n\n"
            "3. **待确认事项**：需要你决定时才出现\n\n"
            "4. **最终交付**：正式测验及技术报告"
        )
        latest = st.session_state.sjt_latest_checkpoint
        if latest is None and st.button(
            "查找未完成任务",
            use_container_width=True,
        ):
            with st.spinner("正在检查历史任务……"):
                try:
                    latest = find_latest_resumable_checkpoint(
                        DEFAULT_CHECKPOINT_ROOT
                    )
                    st.session_state.sjt_latest_checkpoint = latest or False
                except ValueError as exc:
                    st.session_state.sjt_latest_checkpoint = False
                    st.caption(f"已有检查点不可用：{exc}")
        if latest is not None:
            if latest is False:
                st.caption("没有找到可继续的任务。")
                return
            st.divider()
            state = latest["state"]
            st.caption("检测到一项未完成任务")
            st.write(
                f"**{phase_label(state.get('current_phase'))}** · "
                f"{len(state.get('item_pool') or [])} 道候选题"
            )
            if st.button(
                "继续上次任务",
                use_container_width=True,
            ):
                _resume_latest_run()
                st.rerun()


def _render_specification(specification: object) -> None:
    if not isinstance(specification, Mapping):
        st.info("尚未形成测验规格。")
        return
    rows = []
    for field, label in SPECIFICATION_LABELS.items():
        value = specification.get(field)
        if value is not None and value != []:
            rendered = (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
            )
            rows.append({"项目": label, "内容": rendered})
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_item(item: object) -> None:
    if not isinstance(item, Mapping):
        st.info("暂无题目内容。")
        return
    item_id = item.get("item_id")
    if item_id:
        st.caption(f"题目编号：{item_id}")
    st.markdown(f"**情境**\n\n{item.get('scenario', '未提供')}")
    stem = item.get("stem")
    if stem:
        st.markdown(f"**问题**\n\n{stem}")
    options = item.get("options") or []
    if options:
        st.markdown("**选项**")
        for index, option in enumerate(options, 1):
            if isinstance(option, Mapping):
                label = option.get("label") or chr(64 + index)
                text = (
                    option.get("text")
                    or option.get("content")
                    or str(dict(option))
                )
                st.write(f"{label}. {text}")
            else:
                st.write(f"{index}. {option}")
    with st.expander("查看题目结构与计分信息"):
        st.json(dict(item), expanded=False)


def _render_blueprint(blueprint: object) -> None:
    if not isinstance(blueprint, Mapping):
        st.info("暂无蓝图内容。")
        return
    for field in ("title", "summary", "rationale"):
        if blueprint.get(field):
            st.write(blueprint[field])
    is_fixed_blueprint = blueprint.get("version") == 6
    profile = blueprint.get("construct_profile_snapshot")
    if is_fixed_blueprint and isinstance(profile, Mapping):
        st.subheader("构念档案")
        st.write(
            f"{profile.get('inventory_name', '')} · "
            f"{profile.get('domain_name', '')} · "
            f"{profile.get('selection_level', '')}"
        )
        facets = [
            {
                "facet": facet.get("facet_name"),
                "definition": facet.get("definition"),
            }
            for facet in profile.get("facets") or []
            if isinstance(facet, Mapping)
        ]
        if facets:
            st.dataframe(facets, hide_index=True, width="stretch")
        st.caption(
            f"程序汇总：计划生成 {planned_generation_count(blueprint)} 题，"
            f"最终保留 {planned_retention_count(blueprint)} 题。"
        )
    cells = blueprint.get("cells") or []
    if cells:
        st.subheader("维度分配")
        st.dataframe(cells, hide_index=True, width="stretch")
    slots = blueprint.get("slots") or []
    if slots:
        st.subheader("固定题号")
        st.dataframe(slots, hide_index=True, width="stretch")
    with st.expander("查看完整细目表数据"):
        st.json(dict(blueprint), expanded=False)


def _render_review(review: object) -> None:
    if not isinstance(review, Mapping):
        st.info("暂无审查内容。")
        return
    decision = (
        review.get("decision")
        or review.get("overall_decision")
        or review.get("status")
    )
    if decision:
        st.write(f"**审查结论：** {decision}")
    if review.get("summary"):
        st.write(review["summary"])
    issues = list(review.get("issues") or [])
    issues.extend(review.get("findings") or [])
    for section_name in (
        "construct_review",
        "item_skeleton_review",
        "content_review",
    ):
        section = review.get(section_name)
        if isinstance(section, Mapping):
            issues.extend(section.get("issues") or [])
    for issue in issues:
        if isinstance(issue, Mapping):
            message = (
                issue.get("description")
                or issue.get("issue")
                or issue.get("problem")
                or str(dict(issue))
            )
            if issue.get("severity") == "blocking":
                st.error(message)
            else:
                st.warning(message)
        else:
            st.warning(str(issue))
    tasks = review.get("repair_tasks") or []
    if tasks:
        with st.expander("查看程序派生修复任务"):
            st.json(tasks, expanded=False)
    with st.expander("查看完整审查记录"):
        st.json(dict(review), expanded=False)


def _render_generic(value: object) -> None:
    if isinstance(value, list) and value and all(
        isinstance(item, Mapping) for item in value
    ):
        st.dataframe(value, hide_index=True, use_container_width=True)
    elif isinstance(value, Mapping):
        st.json(dict(value), expanded=False)
    else:
        st.write(value)


def _render_content(content: Mapping[str, Any]) -> None:
    for field, value in content.items():
        if field == "test_specification":
            _render_specification(value)
        elif field == "current_item":
            _render_item(value)
        elif field == "blueprint":
            _render_blueprint(value)
        elif field in {
            "blueprint_review",
            "current_item_review",
            "test_review_result",
        }:
            _render_review(value)
        else:
            label = CONTENT_LABELS.get(field, field.replace("_", " "))
            st.markdown(f"**{label}**")
            _render_generic(value)


def _render_timeline() -> None:
    entries = st.session_state.sjt_timeline
    st.subheader("开发过程")
    visible = [
        entry
        for entry in entries
        if (
            (entry.get("event") or {}).get("event_type") == "failed"
            or (
                entry.get("node") in USER_VISIBLE_TIMELINE_NODES
                and (entry.get("event") or {}).get("event_type")
                in {"completed", "waiting"}
            )
        )
    ]
    if not visible:
        st.info("流程启动后，各阶段的输出会依次出现在这里。")
        return

    for index, entry in enumerate(visible):
        event = entry["event"]
        event_type = event.get("event_type")
        icon = "✅" if event_type == "completed" else "⚠️"
        title = f"{icon} {action_label(event.get('action'))}"
        duration = event.get("duration_ms")
        if isinstance(duration, (int, float)):
            title += f" · {duration / 1000:.1f}s"
        expanded = index >= max(0, len(visible) - 2)
        with st.expander(title, expanded=expanded):
            if event.get("reason"):
                st.caption(event["reason"])
            if event.get("error"):
                st.error(event["error"])
            content = entry.get("content") or {}
            if content:
                _render_content(content)
            else:
                st.write("该步骤已完成，没有新增需要展示的业务内容。")

    if st.session_state.sjt_runtime_events:
        with st.expander("运行细节与重试记录"):
            for event in st.session_state.sjt_runtime_events:
                st.write(f"• {_runtime_message(event)}")


def _submit_decision(decision: dict[str, Any]) -> None:
    _advance(Command(resume=decision))
    st.rerun()


def _render_requirement_decision(payload: Mapping[str, Any]) -> None:
    proposed = payload.get("proposed_update") or {}
    if payload.get("summary"):
        st.write(payload["summary"])
    if payload.get("validation_error"):
        st.error(payload["validation_error"])
    st.markdown("**当前测验规格**")
    candidate_specification = proposed.get("test_specification")
    _render_specification(candidate_specification)
    if isinstance(candidate_specification, Mapping):
        try:
            resolved_profile = resolve_specification_profile(
                candidate_specification
            )
        except ValueError:
            resolved_profile = None
        if resolved_profile is not None:
            st.info(
                "构念库解析："
                f"{resolved_profile['inventory_name']} / "
                f"{resolved_profile['domain_name']} / "
                f"{resolved_profile['selection_level']}，"
                f"{len(resolved_profile['facets'])} 个 facet"
            )
    questions = payload.get("questions") or []
    if questions:
        st.markdown("**还需要你确认**")
        for question in questions:
            text = question.get("text") if isinstance(question, Mapping) else question
            st.write(f"• {text}")
    suggestions = payload.get("suggestions") or []
    if suggestions:
        with st.expander("查看系统建议", expanded=True):
            for suggestion in suggestions:
                if isinstance(suggestion, Mapping):
                    st.write(f"**{suggestion.get('field', '字段')}**")
                    if suggestion.get("reason"):
                        st.caption(suggestion["reason"])

    available = set(payload.get("available_decisions") or [])
    choices = {}
    if "confirm" in available:
        choices["确认规格并继续"] = "confirm"
    if "accept_suggestions" in available:
        choices["接受系统建议"] = "accept_suggestions"
    choices["补充或修改需求"] = (
        "revise" if "confirm" in available else "answer"
    )
    choices["停止任务"] = "stop"

    with st.form("requirement_decision"):
        label = st.radio("下一步", list(choices), horizontal=True)
        feedback = st.text_area(
            "补充说明",
            placeholder="选择补充或修改时，请在这里说明。",
        )
        submitted = st.form_submit_button(
            "提交并继续",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        decision = choices[label]
        if decision in {"answer", "revise"} and not feedback.strip():
            st.error("请填写需要补充或修改的内容。")
            return
        _submit_decision(
            {
                "decision": decision,
                "feedback": feedback.strip() or None,
            }
        )


def _render_mode_selection(payload: Mapping[str, Any]) -> None:
    modes = payload.get("modes") or []
    labels = {
        str(mode.get("label") or mode.get("mode")): mode.get("mode")
        for mode in modes
        if isinstance(mode, Mapping)
    }
    if not labels:
        labels = {"逐题人工确认": "manual", "自动开发并集中查看": "automatic"}
    with st.form("mode_selection"):
        selected = st.radio("题目开发方式", list(labels))
        for mode in modes:
            if isinstance(mode, Mapping) and mode.get("description"):
                st.caption(
                    f"{mode.get('label', mode.get('mode'))}："
                    f"{mode['description']}"
                )
        submitted = st.form_submit_button(
            "采用此模式",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        _submit_decision({"mode": labels[selected]})


def _render_sample_selection(payload: Mapping[str, Any]) -> None:
    recommendations = payload.get("recommendations") or []
    options = {}
    for option in recommendations:
        if isinstance(option, Mapping):
            label = str(option.get("label") or option.get("sample_size"))
            if option.get("recommended"):
                label += "（推荐）"
            options[label] = int(option.get("sample_size", 0))
    options["自定义"] = None

    with st.form("sample_selection"):
        selected = st.radio("虚拟被试规模", list(options))
        custom_size = st.number_input(
            "自定义人数",
            min_value=1,
            value=int(payload.get("recommended_sample_size") or 100),
            disabled=options[selected] is not None,
        )
        with st.expander("并发与复现设置"):
            seed = st.number_input(
                "随机种子",
                min_value=0,
                value=int(payload.get("default_seed", 7)),
            )
            concurrency = st.number_input(
                "最大并发",
                min_value=1,
                max_value=int(payload.get("max_allowed_concurrency", 20)),
                value=int(payload.get("default_max_concurrency", 5)),
            )
            retries = st.number_input(
                "失败重试次数",
                min_value=0,
                max_value=10,
                value=int(payload.get("default_max_retries", 2)),
            )
        submitted = st.form_submit_button(
            "开始虚拟作答",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        sample_size = options[selected] or int(custom_size)
        _submit_decision(
            {
                "sample_size": sample_size,
                "seed": int(seed),
                "max_concurrency": int(concurrency),
                "max_retries": int(retries),
            }
        )


def _render_psychometric_repair_confirmation(payload: Mapping[str, Any]) -> None:
    """Show all confirmed edits for one item before the single re-test."""

    st.markdown(f"**题目编号**：{payload.get('item_id', '?')}")
    item = payload.get("item")
    if isinstance(item, Mapping):
        _render_item(item)
    diagnosis = payload.get("diagnosis") or {}
    if isinstance(diagnosis, Mapping) and diagnosis.get("summary"):
        st.write(diagnosis["summary"])
    candidates = diagnosis.get("candidate_diagnoses") if isinstance(diagnosis, Mapping) else []
    if candidates:
        with st.expander("查看诊断候选", expanded=True):
            st.json(candidates, expanded=False)
    tasks = diagnosis.get("repair_tasks") if isinstance(diagnosis, Mapping) else []
    if tasks:
        st.markdown("**本题将逐条执行的修改任务**")
        st.json(tasks, expanded=False)
    with st.form("psychometric_repair_confirmation"):
        selected = st.radio(
            "返修决策",
            [
                "确认全部任务，完成后统一重测",
                "跳过本题修改并保留警告",
                "停止本次运行",
            ],
        )
        submitted = st.form_submit_button(
            "提交决策",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        decisions = {
            "确认全部任务，完成后统一重测": "approve",
            "跳过本题修改并保留警告": "skip",
            "停止本次运行": "stop",
        }
        _submit_decision({"decision": decisions[selected]})


def _render_generic_approval(payload: Mapping[str, Any]) -> None:
    if payload.get("summary"):
        st.write(payload["summary"])
    if payload.get("validation_error"):
        st.error(payload["validation_error"])
    proposed = payload.get("proposed_update") or {}
    action = str(payload.get("action") or "unknown")
    content = {}
    if isinstance(proposed, Mapping):
        content = extract_action_content(action, proposed)
        if not content:
            content = dict(proposed)
    if content:
        _render_content(content)

    choices = {
        "通过并继续": "approve",
        "提出修改意见": "regenerate",
        "停止任务": "stop",
    }
    with st.form("generic_approval"):
        selected = st.radio("你的决定", list(choices), horizontal=True)
        feedback = st.text_area(
            "修改意见",
            placeholder="需要重新生成时，请说明具体要改什么。",
        )
        submitted = st.form_submit_button(
            "提交决定",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        decision = choices[selected]
        if decision == "regenerate" and not feedback.strip():
            st.error("请填写具体修改意见。")
            return
        _submit_decision(
            {
                "decision": decision,
                "feedback": feedback.strip() or None,
                "state_patch": None,
            }
        )


def _render_interrupt() -> None:
    payload = st.session_state.sjt_interrupt
    if not isinstance(payload, Mapping):
        return
    st.subheader("需要你的确认")
    with st.container(border=True):
        interaction_type = payload.get("type")
        if interaction_type == "requirement_confirmation":
            _render_requirement_decision(payload)
        elif interaction_type == "item_development_mode_selection":
            _render_mode_selection(payload)
        elif interaction_type == "virtual_sample_selection":
            _render_sample_selection(payload)
        elif interaction_type == "psychometric_repair_confirmation":
            _render_psychometric_repair_confirmation(payload)
        else:
            _render_generic_approval(payload)


def _download_json(label: str, value: object, filename: str) -> None:
    st.download_button(
        label,
        data=json.dumps(value, ensure_ascii=False, indent=2),
        file_name=filename,
        mime="application/json",
        use_container_width=True,
    )


def _render_deliverables(state: Mapping[str, Any]) -> None:
    final_test = state.get("final_test")
    technical = state.get("technical_report")
    virtual = state.get("virtual_respondent_report")
    if not any((final_test, technical, virtual)):
        return
    st.subheader("最终交付")
    tabs = st.tabs(["正式测验", "技术报告", "虚拟被试报告"])
    with tabs[0]:
        if final_test:
            _render_generic(final_test)
            _download_json("下载正式测验", final_test, "final_test.json")
        else:
            st.info("尚未生成。")
    with tabs[1]:
        if technical:
            _render_generic(technical)
            _download_json(
                "下载技术报告",
                technical,
                "technical_report.json",
            )
        else:
            st.info("尚未生成。")
    with tabs[2]:
        if virtual:
            _render_generic(virtual)
            _download_json(
                "下载虚拟被试报告",
                virtual,
                "virtual_respondent_report.json",
            )
        else:
            st.info("尚未生成。")


def _render_active_page(state: Mapping[str, Any]) -> None:
    summary = progress_summary(state)
    top_left, top_right = st.columns([1, 0.25])
    with top_left:
        st.markdown(
            '<div class="sjt-kicker">ACTIVE DEVELOPMENT</div>',
            unsafe_allow_html=True,
        )
        st.title("测验开发工作台")
        st.caption(state.get("user_request", ""))
    with top_right:
        if st.button("新建任务", use_container_width=True):
            _reset_session()
            st.rerun()

    columns = st.columns(4)
    columns[0].metric("当前阶段", summary["phase"])
    columns[1].metric("已执行步骤", summary["steps"])
    columns[2].metric("候选题", summary["candidate_items"])
    columns[3].metric(
        "最终入选",
        summary["selected_items"] or "—",
    )

    if st.session_state.sjt_error:
        st.error(st.session_state.sjt_error)
        if st.button("从当前状态重试", type="primary"):
            _retry_current_run(state)
            st.rerun()

    _render_interrupt()
    _render_deliverables(state)
    st.divider()
    _render_timeline()

    with st.sidebar:
        st.header("任务概览")
        st.write(f"**阶段**：{summary['phase']}")
        st.write(f"**状态**：{summary['status']}")
        st.write(f"**运行编号**：`{state.get('run_id', '—')}`")
        completion = state.get("completion_checks") or {}
        if completion:
            passed = sum(bool(value) for value in completion.values())
            st.progress(
                passed / len(completion),
                text=f"完成条件 {passed}/{len(completion)}",
            )
        unmet = state.get("unmet_completion_conditions") or []
        if unmet:
            with st.expander("尚未满足的完成条件"):
                for condition in unmet:
                    st.write(f"• {condition}")
        with st.expander("当前完整状态（调试）"):
            st.json(dict(state), expanded=False)


def main() -> None:
    st.set_page_config(
        page_title="SJT 测验开发工作台",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _apply_page_style()
    _initialize_session()
    state = st.session_state.sjt_state
    if isinstance(state, Mapping):
        _render_active_page(state)
    else:
        _render_start_page()
