"""Normalized presentation model for one virtual psychometric round."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

from sjt_system.evaluation.psychometrics import (
    CITC_REVISION_THRESHOLD,
    CROSS_DOMAIN_VTS_THRESHOLD,
    SAME_DOMAIN_VTS_THRESHOLD,
    TARGET_RHO_THRESHOLD,
)


GATE_ORDER = (
    "citc_pass",
    "target_rho_pass",
    "same_domain_vts_pass",
    "cross_domain_vts_pass",
)

_GATE_LABELS = {
    "citc_pass": "CITC",
    "target_rho_pass": "目标rho_s",
    "same_domain_vts_pass": "同域VTS",
    "cross_domain_vts_pass": "跨域VTS",
}


def metric_scalar(value: Any, *preferred_keys: str) -> float | None:
    """Extract a finite numeric metric from scalar or structured evidence."""

    if isinstance(value, Mapping):
        keys = preferred_keys or ("value", "r", "rho")
        for key in keys:
            if key in value:
                return metric_scalar(value.get(key), *preferred_keys)
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _gate_rows(statistics: Mapping[str, Any]) -> list[dict[str, Any]]:
    quality = statistics.get("quality_evaluation") or {}
    qualification = statistics.get("qualification") or {}
    citc = quality.get("facet_citc") or {}
    specificity = quality.get("virtual_target_specificity") or {}
    target = specificity.get("rho_target") or specificity.get("target_spearman") or {}
    same_domain = specificity.get("same_domain_non_target") or {}
    cross_domain = specificity.get("cross_domain_non_target") or {}
    values = {
        "citc_pass": citc.get("r"),
        "target_rho_pass": target.get("rho"),
        "same_domain_vts_pass": same_domain.get("specificity_margin"),
        "cross_domain_vts_pass": cross_domain.get("specificity_margin"),
    }
    thresholds = {
        "citc_pass": CITC_REVISION_THRESHOLD,
        "target_rho_pass": TARGET_RHO_THRESHOLD,
        "same_domain_vts_pass": SAME_DOMAIN_VTS_THRESHOLD,
        "cross_domain_vts_pass": CROSS_DOMAIN_VTS_THRESHOLD,
    }
    return [
        {
            "gate_id": gate_id,
            "label": _GATE_LABELS[gate_id],
            "value": values[gate_id],
            "threshold": thresholds[gate_id],
            "operator": ">=",
            "passes": qualification.get(gate_id) is True,
            "estimable": values[gate_id] is not None,
            "filtering_authority": True,
        }
        for gate_id in GATE_ORDER
    ]


def _contaminant(group: Mapping[str, Any]) -> dict[str, Any]:
    facet = group.get("selected_non_target_facet") or group.get("largest_non_target_facet") or group
    selected_rho = group.get("max_non_target_rho")
    if selected_rho is None:
        selected_rho = group.get("largest_non_target_rho")
    return {
        "dimension_id": group.get("largest_non_target_dimension_id") or group.get("dimension_id"),
        "facet_name": group.get("largest_non_target_facet_name") or group.get("facet_name"),
        "facet_name_en": facet.get("facet_name_en"),
        "domain_id": group.get("largest_non_target_domain_id"),
        "domain_name": facet.get("domain_name"),
        "definition": facet.get("definition"),
        "high_behavior": facet.get("high_behavior"),
        "low_behavior": facet.get("low_behavior"),
        "group_id": group.get("selected_non_target_group_id") or facet.get("group_id"),
        "condition_id": group.get("selected_non_target_condition_id") or facet.get("condition_id"),
        "rho": selected_rho,
        "signed_rho": selected_rho,
        "non_target_spearman": group.get("non_target_spearman") or [],
        "vts": group.get("specificity_margin"),
        "threshold": group.get("margin_threshold"),
    }


def _condition_rows(statistics: Mapping[str, Any]) -> list[dict[str, Any]]:
    quality = statistics.get("quality_evaluation") or {}
    rows: list[dict[str, Any]] = []
    for condition_id, metric in (quality.get("per_condition_metrics") or {}).items():
        if not isinstance(metric, Mapping):
            continue
        # Matched-condition analysis stores the structured value under
        # ``citc``; older/other producers may use ``facet_citc``. Prefer the
        # first shape that actually contains a numeric correlation.
        citc_value = metric_scalar(metric.get("facet_citc"), "r", "value")
        if citc_value is None:
            citc_value = metric_scalar(metric.get("citc"), "r", "value")
        rows.append(
            {
                "condition_id": str(condition_id),
                "arm_id": metric.get("arm_id") or ("target" if condition_id == "target" else str(condition_id).split("__", 1)[0]),
                "group_id": metric.get("group_id") or ("target" if condition_id == "target" else str(condition_id).split("__", 1)[-1]),
                "filtering_authority": condition_id == "target",
                "citc": citc_value,
                "rho": metric_scalar(metric.get("rho"), "rho", "value"),
            }
        )
    return rows


def _option_facet_mean_rows(
    item_id: str,
    diagnostics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in diagnostics.get("by_condition") or []:
        if not isinstance(condition, Mapping):
            continue
        for option in condition.get("options") or []:
            if not isinstance(option, Mapping):
                continue
            rows.append(
                {
                    "item_id": item_id,
                    "condition_id": condition.get("condition_id"),
                    "construct_role": condition.get("construct_role"),
                    "facet_id": condition.get("facet_id"),
                    "facet_name": condition.get("facet_name"),
                    "option_id": option.get("option_id"),
                    "option_score": option.get("score"),
                    "n": option.get("selection_count"),
                    "facet_mean": option.get("facet_mean"),
                    "facet_standard_error": option.get("facet_standard_error"),
                    "filtering_authority": False,
                }
            )
    return rows


def _option_score_comparison_rows(
    item_id: str,
    diagnostics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return one option-aligned row across all three matched conditions."""

    return [
        {
            "item_id": item_id,
            "option_id": row.get("option_id"),
            "option_score": row.get("score"),
            "target_n": row.get("target_n"),
            "target_mean_score": row.get("target_mean_score"),
            "target_standard_error": row.get("target_standard_error"),
            "same_domain_n": row.get("same_domain_n"),
            "same_domain_mean_score": row.get("same_domain_mean_score"),
            "same_domain_standard_error": row.get("same_domain_standard_error"),
            "cross_domain_n": row.get("cross_domain_n"),
            "cross_domain_mean_score": row.get("cross_domain_mean_score"),
            "cross_domain_standard_error": row.get("cross_domain_standard_error"),
            "same_domain_groups": row.get("same_domain_groups") or [],
            "cross_domain_groups": row.get("cross_domain_groups") or [],
            "filtering_authority": False,
        }
        for row in diagnostics.get("option_score_comparisons") or []
        if isinstance(row, Mapping)
    ]


def build_psychometric_round_result(state: Mapping[str, Any]) -> dict[str, Any]:
    """Create one canonical result used by all user-facing surfaces."""

    statistics_by_item = state.get("item_statistics") or {}
    locked_versions = state.get("locked_retained_item_versions") or {}
    dispositions = state.get("item_final_dispositions") or {}
    item_by_id: dict[str, Mapping[str, Any]] = {}
    for collection in (state.get("item_pool") or [], state.get("frozen_item_bank") or []):
        for item in collection:
            if isinstance(item, Mapping) and item.get("item_id") is not None:
                item_by_id[str(item.get("item_id"))] = item

    items: list[dict[str, Any]] = []
    gate_summary = {
        gate_id: {
            "gate_id": gate_id,
            "label": _GATE_LABELS[gate_id],
            "threshold": threshold,
            "operator": ">=",
            "pass_count": 0,
            "item_count": 0,
            "unestimable_count": 0,
        }
        for gate_id, threshold in (
            ("citc_pass", CITC_REVISION_THRESHOLD),
            ("target_rho_pass", TARGET_RHO_THRESHOLD),
            ("same_domain_vts_pass", SAME_DOMAIN_VTS_THRESHOLD),
            ("cross_domain_vts_pass", CROSS_DOMAIN_VTS_THRESHOLD),
        )
    }
    unestimable_option_panels = 0
    all_option_facet_means: list[dict[str, Any]] = []
    all_option_score_comparisons: list[dict[str, Any]] = []
    for item_id, raw_statistics in statistics_by_item.items():
        if not isinstance(raw_statistics, Mapping):
            continue
        item_id = str(item_id)
        item = item_by_id.get(item_id) or {}
        item_version = item.get("version")
        gates = _gate_rows(raw_statistics)
        failed_thresholds = [
            gate["gate_id"] for gate in gates if gate.get("passes") is not True
        ]
        for gate in gates:
            summary = gate_summary[gate["gate_id"]]
            summary["item_count"] += 1
            if gate.get("passes") is True:
                summary["pass_count"] += 1
            if gate.get("estimable") is not True:
                summary["unestimable_count"] += 1

        qualification = raw_statistics.get("qualification") or {}
        qualified = qualification.get("qualified") is True
        locked = locked_versions.get(item_id) == item_version
        disposition = dispositions.get(item_id) or {}
        if disposition.get("status") == "pending_sme_review":
            status = "pending_sme_review"
        elif disposition.get("status") == "eliminated":
            status = "eliminated"
        elif locked and failed_thresholds:
            status = "qualified_locked_warning"
        elif locked:
            status = "qualified_locked"
        elif qualified:
            status = "newly_qualified"
        else:
            status = "pending_treatment"

        quality = raw_statistics.get("quality_evaluation") or {}
        specificity = quality.get("virtual_target_specificity") or {}
        same_domain = specificity.get("same_domain_non_target") or {}
        cross_domain = specificity.get("cross_domain_non_target") or {}
        option_diagnostics = raw_statistics.get("option_choice_diagnostics") or {}
        option_facet_means = _option_facet_mean_rows(
            item_id,
            option_diagnostics,
        )
        all_option_facet_means.extend(option_facet_means)
        option_score_comparisons = _option_score_comparison_rows(
            item_id,
            option_diagnostics,
        )
        all_option_score_comparisons.extend(option_score_comparisons)
        gradient = raw_statistics.get("target_option_gradient") or {}
        unestimable_option_panels += int(gradient.get("estimable") is not True)
        items.append(
            {
                "item_id": item_id,
                "item_version": item_version,
                "status": status,
                "qualified": qualified,
                "failed_thresholds": failed_thresholds,
                "gates": gates,
                "max_contaminants": {
                    "same_domain": _contaminant(same_domain),
                    "cross_domain": _contaminant(cross_domain),
                },
                "per_condition_metrics": _condition_rows(raw_statistics),
                "target_option_gradient": deepcopy(raw_statistics.get("target_option_gradient") or {}),
                "option_choice_diagnostics": deepcopy(dict(option_diagnostics)),
                "option_facet_means": option_facet_means,
                "option_score_comparisons": option_score_comparisons,
                "arm_difference_diagnostics": deepcopy(
                    dict(
                        option_diagnostics.get("arm_difference_diagnostics")
                        or {}
                    )
                ),
                "item": deepcopy(dict(item)),
                "diagnostic_flags": deepcopy(quality.get("diagnostic_flags") or []),
            }
        )

    items.sort(key=lambda row: row["item_id"])
    pending = [row for row in items if row["status"] == "pending_treatment"]
    newly_qualified = [row for row in items if row["status"] == "newly_qualified"]
    locked = [
        row
        for row in items
        if row["status"] in {"qualified_locked", "qualified_locked_warning"}
    ]
    monitoring = [
        row for row in items if row["status"] == "qualified_locked_warning"
    ]
    unestimable_metric_count = sum(
        row["unestimable_count"] for row in gate_summary.values()
    )
    return {
        "schema_version": 2,
        "analysis_round": int(state.get("psychometric_analysis_round") or 0),
        "conditioning": {
            "variable": "condition_id",
            "method": "matched_facet_arms",
            "rank_method": None,
        },
        "summary": {
            "item_count": len(items),
            "newly_qualified_count": len(newly_qualified),
            "pending_treatment_count": len(pending),
            "qualified_locked_count": len(locked),
            "monitoring_warning_count": len(monitoring),
            "unestimable_metric_count": unestimable_metric_count,
            "unestimable_option_panel_count": unestimable_option_panels,
        },
        "gate_summary": [gate_summary[gate_id] for gate_id in GATE_ORDER],
        "items": items,
        "pending_items": pending,
        "newly_qualified_items": newly_qualified,
        "locked_items": locked,
        "monitoring_warnings": monitoring,
        "option_facet_means": all_option_facet_means,
        "option_score_comparisons": all_option_score_comparisons,
        "condition_score_diagnostics": deepcopy(
            (state.get("virtual_sample_config") or {}).get(
                "generation_diagnostics"
            )
            or {}
        ),
        "evidence_scope": "exploratory_virtual_screening_evidence",
    }
