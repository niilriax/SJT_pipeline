"""Generate final test materials and technical reports."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from sjt_system.state import PSJTState
from sjt_system.delivery.lifecycle import evaluate_completion
from sjt_system.runtime.io import (
    write_json_atomic as _write_json_atomic,
    write_text_atomic as _write_text_atomic,
)
from sjt_system.runtime.trace import utc_timestamp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final_reports"
EVIDENCE_NOTICE = "开发期虚拟证据，不代表已经建立真实被试效度"


def _item_statuses(state: Mapping[str, Any]) -> dict[str, str]:
    selected = {
        item.get("item_id")
        for item in state.get("selected_items") or []
        if isinstance(item, Mapping)
    }
    reserve = {
        item.get("item_id")
        for item in state.get("reserve_items") or []
        if isinstance(item, Mapping)
    }
    deferred = {
        item_id
        for item_id in (
            (state.get("selection_results") or {}).get(
                "deferred_revision_item_ids"
            )
            or []
        )
        if item_id
    }
    removed = {
        item.get("item_id")
        for item in state.get("removed_items") or []
        if isinstance(item, Mapping)
    }
    statuses: dict[str, str] = {}
    for item_id in removed:
        if item_id:
            statuses[str(item_id)] = "removed"
    for item_id in deferred:
        statuses[str(item_id)] = "deferred_revision"
    for item_id in reserve:
        if item_id:
            statuses[str(item_id)] = "reserve"
    for item_id in selected:
        if item_id:
            statuses[str(item_id)] = "selected"
    return statuses


def _display_value(value: Any, *, fallback: str = "证据不足") -> str:
    if value is None:
        return fallback
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _repair_evidence(event: Mapping[str, Any]) -> str:
    baseline = event.get("baseline_metrics") or {}
    corrected = (
        baseline.get("facet_corrected_item_total_correlation")
        or baseline.get("corrected_item_total_correlation")
        or {}
    )
    return f"分面内CITC={_display_value(corrected.get('r'))}"


def _technical_report_markdown(
    technical_report: Mapping[str, Any],
) -> str:
    test_quality = technical_report.get("test_quality") or {}
    reliability = test_quality.get("reliability") or {}
    validity = test_quality.get("validity") or {}
    convergent = validity.get("convergent") or {}
    discriminant = validity.get("discriminant") or {}
    selection = technical_report.get("selection_results") or {}
    optimizer = (
        technical_report.get("blueprint_coverage") or {}
    ).get("optimizer") or {}
    selected_metrics = optimizer.get("selected_metrics") or {}
    developmental_status = technical_report.get("development_status")
    provisional_item_ids = technical_report.get("provisional_item_ids") or []
    repair_history = technical_report.get("psychometric_repair_history") or []
    selection_history = (
        technical_report.get("psychometric_selection_history") or []
    )
    test_statistics = technical_report.get("test_statistics") or {}
    facet_statistics = test_statistics.get("dimensions") or {}
    item_statistics = technical_report.get("item_statistics") or {}
    facet_lines: list[str] = ["## 分面测量结果", ""]
    for facet_id, facet in facet_statistics.items():
        facet_lines.extend(
            [
                f"### {facet_id}",
                "",
                f"- 题数：{facet.get('item_count', 0)}",
                "- Cronbach α："
                + _display_value(facet.get("cronbach_alpha")),
                "",
                "| 题目 | 分面内CITC | 难度 | 有效选项数 | 结果 |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for item_id in facet.get("item_ids") or []:
            item = item_statistics.get(item_id) or {}
            quality = item.get("quality_evaluation") or {}
            citc = item.get("facet_corrected_item_total_correlation") or {}
            facet_lines.append(
                f"| {item_id} | {_display_value(citc.get('r'))} | "
                f"{_display_value(item.get('difficulty'))} | "
                f"{_display_value(quality.get('effective_option_count'))} | "
                f"{_display_value(quality.get('recommendation'))} |"
            )
        facet_lines.append("")
    lines = [
        "# PSJT开发技术报告",
        "",
        f"> {EVIDENCE_NOTICE}",
        "",
        "## 开发概况",
        "",
        f"- 题库版本：{technical_report.get('item_bank_version')}",
        f"- 正式题数：{technical_report.get('selected_item_count')}",
        f"- 备选题数：{technical_report.get('reserve_item_count')}",
        (
            "- 交付等级：开发版（含未达到正式质量门槛的暂定题目）"
            if developmental_status == "developmental"
            else "- 交付等级：标准候选版"
        ),
        f"- 暂定题数：{len(provisional_item_ids)}",
        (
            "- 待后续返修题数："
            f"{technical_report.get('deferred_revision_count', 0)}"
        ),
        "- 心理骨架审核：按固定槽位逐题独立审核",
        "- 心理测量处理：结构化诊断、定向返修、重新模拟与重新分析",
        "",
        "## 候选题库整体测量结果",
        "",
        f"- 开发状态：{_display_value(test_quality.get('overall_status'))}",
        (
            "- 使用状态："
            + _display_value(test_quality.get("operational_use_status"))
        ),
        (
            "- Cronbach α："
            + _display_value(reliability.get("cronbach_alpha"))
        ),
        (
            "- 信度等级："
            + _display_value(reliability.get("overall_grade"))
        ),
        (
            "- 目标维度相关："
            + _display_value(convergent.get("rho"))
            + f"（{_display_value(convergent.get('target'), fallback='未指定')}）"
        ),
        (
            "- 最大非目标相关："
            + _display_value(discriminant.get("largest_non_target_rho"))
            + "；区分差值："
            + _display_value(discriminant.get("target_margin"))
        ),
        (
            "- 结论："
            + _display_value(test_quality.get("interpretation"))
        ),
        "",
        *facet_lines,
        "## 最终正式组合的开发期指标",
        "",
        (
            "- 最差人格模式目标相关："
            + _display_value(selected_metrics.get("worst_case_target_rho"))
        ),
        (
            "- 最差人格模式区分差值："
            + _display_value(
                selected_metrics.get("worst_case_discriminant_margin")
            )
        ),
        (
            "- 最差人格模式 Cronbach α："
            + _display_value(
                selected_metrics.get("worst_case_cronbach_alpha")
            )
        ),
        (
            "- 题间冗余："
            + _display_value(
                selected_metrics.get(
                    "worst_case_inter_item_redundancy"
                )
            )
            + "；语义冗余："
            + _display_value(selected_metrics.get("semantic_redundancy"))
        ),
        "",
        "## 题目筛选与组卷",
        "",
        (
            "- 筛选状态："
            + _display_value(selection.get("status"), fallback="未执行")
        ),
        (
            "- 自动淘汰是否抑制："
            + ("是" if selection.get("automatic_removal_suppressed") else "否")
        ),
        (
            "- 组合优化："
            + _display_value(optimizer.get("method"), fallback="未执行")
            + f"；候选组合={optimizer.get('combination_count', 0)}"
        ),
        (
            f"- 心理测量分析轮数：{len(selection_history)}；"
            f"完成返修事件：{len(repair_history)}；"
            f"重新组卷轮数：{technical_report.get('reassembly_round', 0)}"
        ),
        "",
        "## 题目去向",
        "",
        "| 题目 | 最终状态 | 统计建议 | 原因 |",
        "|---|---|---|---|",
        *[
            "| "
            + " | ".join(
                str(row.get(key) or "未记录").replace("|", "｜")
                for key in (
                    "item_id",
                    "final_status",
                    "recommendation",
                    "reason",
                )
            )
            + " |"
            for row in technical_report.get("item_decisions") or []
        ],
        "",
        "## 心理测量返修记录",
        "",
        "| 题目 | 轮次 | 动作 | 结果 | 返修前证据 |",
        "|---|---:|---|---|---|",
        *[
            "| "
            + " | ".join(
                [
                    str(event.get("item_id") or "未记录").replace("|", "｜"),
                    str(event.get("revision_round") or "未记录"),
                    str(event.get("action") or "未记录").replace("|", "｜"),
                    (
                        "通过"
                        if event.get("event") == "psychometric_item_repaired"
                        else "未通过并退出"
                    ),
                    _repair_evidence(event).replace("|", "｜"),
                ]
            )
            + " |"
            for event in repair_history
            if isinstance(event, Mapping)
        ],
        *(
            ["| 无 | - | - | - | - |"]
            if not repair_history
            else []
        ),
        "",
        "## 证据边界",
        "",
        *(
            [technical_report["context_source_notice"], ""]
            if technical_report.get("context_source_notice")
            else []
        ),
        *(
            [technical_report["construct_registry_notice"], ""]
            if technical_report.get("construct_registry_notice")
            else []
        ),
        EVIDENCE_NOTICE + "。",
        "Neo-FFI和SJT均来自相同虚拟人格输入，相关结果不能替代独立真实效标。",
        "",
    ]
    return "\n".join(lines)


def run_report_generation(
    state: PSJTState,
    *,
    output_root: str | Path = DEFAULT_REPORT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Persist final materials using only evidence already in State."""

    review = state.get("test_review_result")
    assembled = state.get("assembled_test")
    if not isinstance(review, Mapping) or review.get("decision") != "PASS":
        raise ValueError("只有测验级审核 PASS 后才能生成最终报告")
    if not isinstance(assembled, Mapping):
        raise ValueError("生成最终报告前缺少 assembled_test")
    if (
        assembled.get("item_bank_id") != state.get("item_bank_id")
        or assembled.get("item_bank_version")
        != state.get("item_bank_version")
    ):
        raise ValueError("assembled_test 与当前题库版本不一致")

    fingerprint = str(state.get("item_bank_fingerprint") or "unknown")
    output_dir = (
        Path(output_root)
        / str(state["run_id"])
        / f"bank-v{state['item_bank_version']}-{fingerprint[:12]}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    final_test_path = output_dir / "final_test.json"
    item_database_path = output_dir / "item_database.json"
    technical_report_path = output_dir / "technical_report.json"
    technical_markdown_path = output_dir / "technical_report.md"
    virtual_report_path = output_dir / "virtual_respondent_report.json"
    report_manifest_path = output_dir / "report_manifest.json"
    generated_at = utc_timestamp()

    final_test = {
        "schema_version": 1,
        "test_id": assembled.get("test_id"),
        "item_bank_id": state.get("item_bank_id"),
        "item_bank_version": state.get("item_bank_version"),
        "item_bank_fingerprint": state.get("item_bank_fingerprint"),
        "respondent_form": deepcopy(assembled.get("respondent_form")),
        "scoring_key": deepcopy(assembled.get("scoring_key")),
        "blueprint_summary": deepcopy(assembled.get("blueprint_summary")),
        "development_status": (
            "developmental" if assembled.get("provisional") else "standard"
        ),
        "quality_gate": deepcopy(assembled.get("quality_gate")),
        "delivery_warnings": deepcopy(
            (state.get("selection_results") or {}).get("delivery_warnings")
            or []
        ),
        "assembly_files": deepcopy(assembled.get("files")),
        "evidence_notice": EVIDENCE_NOTICE,
        "finalized_at": generated_at,
    }
    statuses = _item_statuses(state)
    item_statistics = state.get("item_statistics") or {}
    database_items = []
    known_ids: set[str] = set()
    for raw_item in state.get("frozen_item_bank") or []:
        if not isinstance(raw_item, Mapping):
            continue
        item = deepcopy(dict(raw_item))
        item_id = str(item.get("item_id"))
        known_ids.add(item_id)
        database_items.append(
            {
                **item,
                "final_status": statuses.get(item_id, "unclassified"),
                "psychometric_statistics": deepcopy(
                    item_statistics.get(item_id)
                ),
                "provisional_quality_flag": deepcopy(
                    (state.get("provisional_item_flags") or {}).get(item_id)
                ),
            }
        )
    for raw_item in state.get("removed_items") or []:
        if not isinstance(raw_item, Mapping):
            continue
        item_id = str(raw_item.get("item_id"))
        if item_id in known_ids:
            continue
        database_items.append(
            {
                **deepcopy(dict(raw_item)),
                "final_status": "removed",
                "psychometric_statistics": None,
            }
        )
    item_database = {
        "schema_version": 1,
        "run_id": state["run_id"],
        "item_bank_id": state.get("item_bank_id"),
        "item_bank_version": state.get("item_bank_version"),
        "items": database_items,
        "generated_at": generated_at,
    }

    measurement_evaluation = (
        (state.get("test_statistics") or {}).get(
            "measurement_evaluation"
        )
        or {}
    )
    decision_index: dict[str, dict[str, Any]] = {}
    for history_entry in state.get("psychometric_selection_history") or []:
        if not isinstance(history_entry, Mapping):
            continue
        recommendations = (
            history_entry.get("effective_recommendations")
            or history_entry.get("source_recommendations")
            or {}
        )
        reasons = history_entry.get("reasons") or {}
        for item_id, recommendation in recommendations.items():
            decision_index[str(item_id)] = {
                "recommendation": recommendation,
                "reason": reasons.get(item_id) or "未记录详细原因",
            }
    current_recommendations = (
        (state.get("selection_results") or {}).get(
            "effective_recommendations"
        )
        or (state.get("selection_results") or {}).get(
            "source_recommendations"
        )
        or {}
    )
    for item_id, recommendation in current_recommendations.items():
        decision_index[str(item_id)] = {
            "recommendation": recommendation,
            "reason": (state.get("selection_reasons") or {}).get(item_id)
            or decision_index.get(str(item_id), {}).get("reason")
            or "未记录详细原因",
        }
    final_dispositions = state.get("item_final_dispositions") or {}
    for item_id, disposition in final_dispositions.items():
        if isinstance(disposition, Mapping):
            decision_index[str(item_id)] = {
                "recommendation": disposition.get("status"),
                "reason": (
                    disposition.get("warning_reason")
                    or (state.get("selection_reasons") or {}).get(item_id)
                    or "accepted"
                ),
            }
    all_decision_ids = list(
        dict.fromkeys(
            [
                *decision_index,
                *statuses,
                *[
                    str(item.get("item_id"))
                    for item in state.get("removed_items") or []
                    if isinstance(item, Mapping) and item.get("item_id")
                ],
            ]
        )
    )
    item_decisions = [
        {
            "item_id": item_id,
            "final_status": (
                (final_dispositions.get(item_id) or {}).get("status")
                or statuses.get(item_id, "unclassified")
            ),
            "recommendation": decision_index.get(item_id, {}).get(
                "recommendation"
            ),
            "reason": decision_index.get(item_id, {}).get("reason"),
        }
        for item_id in all_decision_ids
    ]
    technical_report = {
        "schema_version": 1,
        "run_id": state["run_id"],
        "item_bank_id": state.get("item_bank_id"),
        "item_bank_version": state.get("item_bank_version"),
        "test_specification": deepcopy(state.get("test_specification")),
        "construct_profile_ref": deepcopy(
            (state.get("blueprint") or {}).get("construct_profile_ref")
        ),
        "construct_profile": deepcopy(state.get("construct_profile")),
        "blueprint": deepcopy(state.get("blueprint")),
        "skeleton_reviews": deepcopy(state.get("skeleton_reviews") or {}),
        "context_source_notice": (
            "情境类别由模型根据目标人群与使用需求合成，仅作为内容设计候选；"
            "不代表访谈资料、实证频率或效标证据。"
            if state.get("construct_profile") is not None
            else None
        ),
        "construct_registry_notice": (
            "本次构念语义来自版本化种子注册表；其来源边界和本地化表述仍应"
            "在正式使用前由心理测量领域专家审定。"
            if isinstance(state.get("construct_profile"), Mapping)
            and state["construct_profile"].get("review_status")
            == "versioned_seed"
            else None
        ),
        "selected_item_count": len(state.get("selected_items") or []),
        "reserve_item_count": len(state.get("reserve_items") or []),
        "deferred_revision_count": len(
            state.get("items_deferred_for_revision") or []
        ),
        "removed_item_count": len(state.get("removed_items") or []),
        "development_status": (
            "developmental" if assembled.get("provisional") else "standard"
        ),
        "provisional_item_ids": deepcopy(
            (state.get("selection_results") or {}).get("provisional_item_ids")
            or []
        ),
        "provisional_item_flags": deepcopy(
            state.get("provisional_item_flags") or {}
        ),
        "test_statistics": deepcopy(state.get("test_statistics")),
        "item_statistics": deepcopy(state.get("item_statistics") or {}),
        "test_quality": deepcopy(measurement_evaluation),
        "selection_results": deepcopy(state.get("selection_results")),
        "item_final_dispositions": deepcopy(
            state.get("item_final_dispositions") or {}
        ),
        "locked_retained_item_versions": deepcopy(
            state.get("locked_retained_item_versions") or {}
        ),
        "best_assembly_candidate": deepcopy(
            state.get("best_assembly_candidate")
        ),
        "blueprint_coverage": deepcopy(state.get("blueprint_coverage")),
        "test_review_result": deepcopy(state.get("test_review_result")),
        "psychometric_selection_history": deepcopy(
            state.get("psychometric_selection_history") or []
        ),
        "psychometric_repair_history": deepcopy(
            state.get("psychometric_repair_history") or []
        ),
        "item_decisions": item_decisions,
        "reassembly_round": state.get("reassembly_round", 0),
        "test_revision_history": deepcopy(
            state.get("test_revision_history") or []
        ),
        "evidence_notice": EVIDENCE_NOTICE,
        "generated_at": generated_at,
    }
    virtual_report = {
        "schema_version": 1,
        "run_id": state["run_id"],
        "sample_config": deepcopy(state.get("virtual_sample_config")),
        "respondent_count": len(state.get("virtual_respondents") or []),
        "response_summary": deepcopy(
            state.get("virtual_response_summary")
        ),
        "response_manifest": state.get("virtual_response_data_ref"),
        "item_bank_id": state.get("virtual_response_item_bank_id"),
        "item_bank_version": state.get(
            "virtual_response_item_bank_version"
        ),
        "interpretation_limitations": [
            EVIDENCE_NOTICE,
            (
                "虚拟SJT和Neo-FFI作答来自相同人格输入，相关性可能受到"
                "共同方法和语义重叠影响。"
            ),
            "本报告不包含真实被试、重测信度或人口学DIF证据。",
        ],
        "generated_at": generated_at,
    }

    _write_json_atomic(final_test_path, final_test)
    _write_json_atomic(item_database_path, item_database)
    _write_json_atomic(technical_report_path, technical_report)
    _write_text_atomic(
        technical_markdown_path,
        _technical_report_markdown(technical_report),
    )
    _write_json_atomic(virtual_report_path, virtual_report)

    proposed_state = {
        **state,
        "final_test": final_test,
        "item_database_ref": str(item_database_path.resolve()),
        "technical_report": technical_report,
        "virtual_respondent_report": virtual_report,
    }
    checks, unmet = evaluate_completion(proposed_state)
    manifest = {
        "schema_version": 1,
        "run_id": state["run_id"],
        "item_bank_id": state.get("item_bank_id"),
        "item_bank_version": state.get("item_bank_version"),
        "generated_at": generated_at,
        "evidence_notice": EVIDENCE_NOTICE,
        "completion_checks": checks,
        "unmet_completion_conditions": unmet,
        "files": {
            "final_test": str(final_test_path.resolve()),
            "item_database": str(item_database_path.resolve()),
            "technical_report": str(technical_report_path.resolve()),
            "technical_report_markdown": str(
                technical_markdown_path.resolve()
            ),
            "virtual_respondent_report": str(
                virtual_report_path.resolve()
            ),
        },
    }
    _write_json_atomic(report_manifest_path, manifest)
    return {
        "state_update": {
            "final_test": {
                **final_test,
                "file_path": str(final_test_path.resolve()),
                "report_manifest_path": str(
                    report_manifest_path.resolve()
                ),
            },
            "item_database_ref": str(item_database_path.resolve()),
            "technical_report": {
                **technical_report,
                "file_path": str(technical_report_path.resolve()),
                "markdown_path": str(
                    technical_markdown_path.resolve()
                ),
            },
            "virtual_respondent_report": {
                **virtual_report,
                "file_path": str(virtual_report_path.resolve()),
            },
            "completion_checks": checks,
            "unmet_completion_conditions": unmet,
        },
        "summary": (
            "最终测验、题库、技术报告和虚拟被试报告已生成；"
            f"未满足完成条件 {len(unmet)} 项。"
        ),
    }
