"""Whole-test metrics for provisional iteration forms.

The item-level screening gates remain the authority for repairing individual
items.  This module evaluates the provisional form assembled at each
development round so the workflow can show whether the complete test is
improving, even while some items are still under treatment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


PLATEAU_DEFAULT_PATIENCE = 2
PLATEAU_DEFAULT_MIN_DELTA = 0.01
TARGET_RECOVERY_DEFAULT_FOLDS = 5
TARGET_RECOVERY_RIDGE_PENALTY = 1.0
FORM_EFFECT_EXTREME_FRACTION = 1.0 / 3.0
VIRTUAL_FORM_ICC_DEFAULT_MINIMUM = 0.80
FORM_QUALITY_EPSILON = 1e-12


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _spearman(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    if len(left) < 3 or len(right) < 3:
        return None
    try:
        result = stats.spearmanr(left, right, nan_policy="omit")
    except Exception:
        return None
    return _number(getattr(result, "statistic", None))


def _cronbach_alpha(frame: pd.DataFrame) -> float | None:
    if frame.shape[1] < 2 or frame.shape[0] < 3:
        return None
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if numeric.shape[0] < 3 or numeric.shape[1] < 2:
        return None
    total_variance = float(numeric.sum(axis=1).var(ddof=1))
    if not np.isfinite(total_variance) or total_variance <= 0:
        return None
    item_variance = float(numeric.var(axis=0, ddof=1).sum())
    alpha = numeric.shape[1] / (numeric.shape[1] - 1) * (
        1.0 - item_variance / total_variance
    )
    return _number(alpha)


def _read_scored_responses(test_statistics: Mapping[str, Any]) -> pd.DataFrame | None:
    output_files = test_statistics.get("output_files") or {}
    if not isinstance(output_files, Mapping):
        return None
    path_value = output_files.get("scored_matched_condition_sjt_responses")
    if not isinstance(path_value, str) or not Path(path_value).is_file():
        return None
    try:
        frame = pd.read_csv(path_value)
    except Exception:
        return None
    required = {
        "condition_id",
        "arm_id",
        "group_id",
        "matched_subject_id",
        "item_id",
        "score",
        "active_score",
    }
    if not required <= set(frame.columns):
        return None
    frame["item_id"] = frame["item_id"].astype(str)
    frame["condition_id"] = frame["condition_id"].astype(str)
    frame["arm_id"] = frame["arm_id"].astype(str)
    frame["group_id"] = frame["group_id"].astype(str)
    frame["matched_subject_id"] = frame["matched_subject_id"].astype(str)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["active_score"] = pd.to_numeric(frame["active_score"], errors="coerce")
    return frame


def _read_scored_target_retests(
    test_statistics: Mapping[str, Any],
) -> pd.DataFrame | None:
    output_files = test_statistics.get("output_files") or {}
    if not isinstance(output_files, Mapping):
        return None
    path_value = output_files.get("scored_target_form_retest_sjt_responses")
    if not isinstance(path_value, str) or not Path(path_value).is_file():
        return None
    try:
        frame = pd.read_csv(path_value)
    except Exception:
        return None
    required = {
        "administration_id",
        "condition_id",
        "matched_subject_id",
        "item_id",
        "score",
    }
    if not required <= set(frame.columns):
        return None
    frame = frame[frame["condition_id"].astype(str) == "target"].copy()
    frame["administration_id"] = pd.to_numeric(
        frame["administration_id"], errors="coerce"
    )
    frame["matched_subject_id"] = frame["matched_subject_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(
        subset=["administration_id", "matched_subject_id", "item_id", "score"]
    )
    return frame if not frame.empty else None


def _read_jsonl(path_value: Any) -> list[dict[str, Any]]:
    if not isinstance(path_value, str) or not Path(path_value).is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        with Path(path_value).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if isinstance(record, dict):
                        records.append(record)
    except Exception:
        return []
    return records


def _neo_dimension_code(domain_id: Any) -> str | None:
    normalized = str(domain_id or "").strip().lower()
    for prefix, code in {
        "openness": "O",
        "conscientiousness": "C",
        "extraversion": "E",
        "agreeableness": "A",
        "neuroticism": "N",
    }.items():
        if normalized == prefix or normalized.startswith(prefix + "_"):
            return code
    return None


def _read_virtual_reference_scores(
    test_statistics: Mapping[str, Any],
    *,
    target_subject_ids: Sequence[str],
) -> dict[str, Any]:
    """Read matched target reference questionnaires for development-only validity."""

    output_files = test_statistics.get("output_files") or {}
    reference_meta = test_statistics.get("reference_questionnaires") or {}
    if not isinstance(output_files, Mapping) or not isinstance(reference_meta, Mapping):
        return {"status": "unavailable", "reason": "整卷统计缺少参照问卷元数据"}
    target_ids = {str(value) for value in target_subject_ids}
    if not target_ids:
        return {"status": "unavailable", "reason": "没有可对齐的 target 被试"}
    scores = pd.DataFrame(index=sorted(target_ids))
    details: dict[str, Any] = {}

    neo_records = [
        record
        for record in _read_jsonl(output_files.get("neo_ffi_responses"))
        if str(record.get("condition_id")) == "target"
        and str(record.get("matched_subject_id")) in target_ids
    ]
    neo_meta = reference_meta.get("neo_ffi")
    if isinstance(neo_meta, Mapping) and neo_records:
        code = _neo_dimension_code(neo_meta.get("target_domain_id"))
        if code is None:
            code = str(neo_meta.get("target_dimension_code") or "") or None
        selected = [
            record
            for record in neo_records
            if str(record.get("dimension_code")) == code
        ]
        if code and selected:
            # Do not construct a DataFrame from the complete model records.
            # Older response files may contain non-tabular metadata fields
            # (lists/dicts), which makes pandas' string dtype conversion fail.
            grouped_scores: dict[str, list[float]] = {}
            for record in selected:
                subject_id = str(record.get("matched_subject_id") or "")
                raw_score = record.get("score")
                if not subject_id or isinstance(raw_score, bool):
                    continue
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(score):
                    grouped_scores.setdefault(subject_id, []).append(score)
            neo_scores = pd.Series(
                {
                    subject_id: float(np.mean(values))
                    for subject_id, values in grouped_scores.items()
                    if values
                },
                dtype="float64",
                name="neo_ffi_target_score",
            )
            scores["neo_ffi_target_score"] = neo_scores
            details["neo_ffi"] = {
                "dimension_code": code,
                "sample_size": int(neo_scores.notna().sum()),
                "item_response_count": int(len(selected)),
            }

    available = [column for column in scores.columns if scores[column].notna().any()]
    if not available:
        return {
            "status": "unavailable",
            "reason": "当前轮次没有完整的 Neo-FFI target 参照分数",
            "details": details,
        }
    return {
        "status": "complete" if len(available) == 2 else "partial",
        "scores": scores[available],
        "details": details,
    }


def prepare_provisional_form_metric_context(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Load round-level form data once for repeated candidate-form scoring."""

    test_statistics = state.get("test_statistics") or {}
    if not isinstance(test_statistics, Mapping):
        return {"responses": None, "reference_result": {"status": "unavailable"}}
    responses = _read_scored_responses(test_statistics)
    if responses is None:
        return {"responses": None, "reference_result": {"status": "unavailable"}}
    condition_item_scores: dict[str, pd.DataFrame] = {}
    condition_active_scores: dict[str, pd.Series] = {}
    condition_metadata: dict[str, dict[str, str]] = {}
    for condition_id, condition in responses.groupby("condition_id", sort=True):
        condition_key = str(condition_id)
        condition_item_scores[condition_key] = condition.pivot_table(
            index="matched_subject_id",
            columns="item_id",
            values="score",
            aggfunc="first",
        ).sort_index()
        condition_active_scores[condition_key] = (
            condition[["matched_subject_id", "active_score"]]
            .drop_duplicates("matched_subject_id")
            .set_index("matched_subject_id")["active_score"]
            .sort_index()
        )
        first = condition.iloc[0]
        condition_metadata[condition_key] = {
            "arm_id": str(first.get("arm_id") or ""),
            "group_id": str(first.get("group_id") or ""),
        }
    target_ids = (
        responses.loc[
            responses["condition_id"] == "target", "matched_subject_id"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    target_scores = condition_item_scores.get("target", pd.DataFrame())
    target_active = condition_active_scores.get(
        "target", pd.Series(dtype="float64")
    )
    group_active: dict[str, dict[str, pd.Series]] = {}
    for condition_id, active in condition_active_scores.items():
        if condition_id == "target":
            continue
        metadata = condition_metadata.get(condition_id) or {}
        group_active.setdefault(str(metadata.get("arm_id") or ""), {})[
            str(metadata.get("group_id") or condition_id)
        ] = active
    target_retest_scores: dict[int, pd.DataFrame] = {}
    target_retests = _read_scored_target_retests(test_statistics)
    if target_retests is not None:
        for administration_id, administration in target_retests.groupby(
            "administration_id", sort=True
        ):
            target_retest_scores[int(administration_id)] = (
                administration.pivot_table(
                    index="matched_subject_id",
                    columns="item_id",
                    values="score",
                    aggfunc="first",
                ).sort_index()
            )
    return {
        "responses": responses,
        "target_scores": target_scores,
        "target_active": target_active,
        "group_active": group_active,
        "condition_item_scores": condition_item_scores,
        "condition_active_scores": condition_active_scores,
        "condition_metadata": condition_metadata,
        "target_retest_scores": target_retest_scores,
        "reference_result": _read_virtual_reference_scores(
            test_statistics,
            target_subject_ids=target_ids,
        ),
    }


def _icc_absolute_agreement_single(scores: pd.DataFrame) -> float | None:
    """Two-way absolute-agreement single-measure ICC, ICC(A,1)."""

    numeric = scores.apply(pd.to_numeric, errors="coerce").dropna(
        axis=0, how="any"
    )
    n_subjects, n_administrations = numeric.shape
    if n_subjects < 3 or n_administrations < 2:
        return None
    values = numeric.to_numpy(dtype=float)
    grand_mean = float(values.mean())
    subject_means = values.mean(axis=1)
    administration_means = values.mean(axis=0)
    ms_subject = n_administrations * float(
        np.square(subject_means - grand_mean).sum()
    ) / float(n_subjects - 1)
    ms_administration = n_subjects * float(
        np.square(administration_means - grand_mean).sum()
    ) / float(n_administrations - 1)
    residual = (
        values
        - subject_means[:, None]
        - administration_means[None, :]
        + grand_mean
    )
    ms_error = float(np.square(residual).sum()) / float(
        (n_subjects - 1) * (n_administrations - 1)
    )
    denominator = (
        ms_subject
        + (n_administrations - 1) * ms_error
        + n_administrations
        * (ms_administration - ms_error)
        / n_subjects
    )
    if not np.isfinite(denominator) or denominator <= 0:
        return None
    return _number((ms_subject - ms_error) / denominator)


def _stable_fold_ids(index: Sequence[Any], fold_count: int) -> np.ndarray:
    """Assign deterministic folds without depending on row order."""

    hashed = np.asarray(
        [
            int.from_bytes(
                sha256(str(value).encode("utf-8")).digest()[:8], "big"
            )
            for value in index
        ],
        dtype=np.uint64,
    )
    order = np.argsort(hashed, kind="stable")
    folds = np.empty(len(order), dtype=int)
    folds[order] = np.arange(len(order), dtype=int) % fold_count
    return folds


def _cross_validated_target_recovery(
    item_scores: pd.DataFrame,
    target_scores: pd.Series,
    *,
    fold_count: int = TARGET_RECOVERY_DEFAULT_FOLDS,
    ridge_penalty: float = TARGET_RECOVERY_RIDGE_PENALTY,
) -> dict[str, Any]:
    """Recover assigned target scores from the complete item-response pattern."""

    aligned = item_scores.join(target_scores.rename("target_score"), how="inner")
    aligned = aligned.apply(pd.to_numeric, errors="coerce").dropna(
        axis=0, how="any"
    )
    n = len(aligned)
    resolved_folds = min(int(fold_count), n)
    if n < 6 or resolved_folds < 2:
        return {
            "status": "unavailable",
            "reason": "交叉验证目标恢复至少需要6名完整虚拟被试",
            "sample_size": n,
        }
    x = aligned[item_scores.columns].to_numpy(dtype=float)
    y = aligned["target_score"].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return {"status": "unavailable", "reason": "目标恢复数据包含非有限值"}
    if float(np.var(y, ddof=1)) <= 0:
        return {"status": "unavailable", "reason": "目标构念分数没有变异"}
    folds = _stable_fold_ids(aligned.index.tolist(), resolved_folds)
    predictions = np.full(n, np.nan, dtype=float)
    for fold in range(resolved_folds):
        test_mask = folds == fold
        train_mask = ~test_mask
        if int(test_mask.sum()) == 0 or int(train_mask.sum()) < 3:
            continue
        train_x = x[train_mask]
        test_x = x[test_mask]
        train_y = y[train_mask]
        x_mean = train_x.mean(axis=0)
        x_sd = train_x.std(axis=0, ddof=1)
        x_sd = np.where(
            np.isfinite(x_sd) & (x_sd > 1e-12), x_sd, 1.0
        )
        train_z = (train_x - x_mean) / x_sd
        test_z = (test_x - x_mean) / x_sd
        y_mean = float(train_y.mean())
        centered_y = train_y - y_mean
        gram = train_z.T @ train_z
        penalty = float(ridge_penalty) * np.eye(gram.shape[0], dtype=float)
        try:
            coefficients = np.linalg.solve(
                gram + penalty, train_z.T @ centered_y
            )
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(gram + penalty) @ (
                train_z.T @ centered_y
            )
        predictions[test_mask] = y_mean + test_z @ coefficients
    if not np.isfinite(predictions).all():
        return {"status": "unavailable", "reason": "交叉验证预测不完整"}
    denominator = float(np.square(y - y.mean()).sum())
    if denominator <= 0:
        return {"status": "unavailable", "reason": "目标构念分数没有变异"}
    r_squared = 1.0 - float(np.square(y - predictions).sum()) / denominator
    prediction_rho = _spearman(predictions.tolist(), y.tolist())
    return {
        "status": "complete",
        "cross_validated_r2": _number(r_squared),
        "prediction_spearman": prediction_rho,
        "sample_size": n,
        "item_count": int(item_scores.shape[1]),
        "fold_count": resolved_folds,
        "model": "ridge_regression_on_complete_item_pattern",
        "ridge_penalty": float(ridge_penalty),
    }


def _extreme_group_effect(
    form_scores: pd.Series,
    active_scores: pd.Series,
    *,
    denominator_sd: float,
) -> dict[str, Any]:
    aligned = pd.concat(
        [form_scores.rename("form_score"), active_scores.rename("active_score")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 6 or not np.isfinite(denominator_sd) or denominator_sd <= 0:
        return {"status": "unavailable", "sample_size": int(len(aligned))}
    low_threshold = float(
        aligned["active_score"].quantile(FORM_EFFECT_EXTREME_FRACTION)
    )
    high_threshold = float(
        aligned["active_score"].quantile(1.0 - FORM_EFFECT_EXTREME_FRACTION)
    )
    low = aligned[aligned["active_score"] <= low_threshold]["form_score"]
    high = aligned[aligned["active_score"] >= high_threshold]["form_score"]
    if len(low) < 2 or len(high) < 2:
        return {"status": "unavailable", "sample_size": int(len(aligned))}
    raw_difference = float(high.mean() - low.mean())
    return {
        "status": "complete",
        "standardized_effect": _number(raw_difference / denominator_sd),
        "raw_score_difference": _number(raw_difference),
        "low_mean": _number(float(low.mean())),
        "high_mean": _number(float(high.mean())),
        "low_n": int(len(low)),
        "high_n": int(len(high)),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "sample_size": int(len(aligned)),
    }


def _construct_selectivity_value(
    target_sensitivity: Any,
    maximum_leakage: Any,
) -> float | None:
    """Return the bounded target-signal share used by whole-form optimization."""

    target = _number(target_sensitivity)
    leakage = _number(maximum_leakage)
    if target is None or leakage is None:
        return None
    # A reversed target effect is not useful target transmission.  Preserve
    # the raw signed effect elsewhere, but give it zero optimization credit.
    target_signal = max(0.0, target)
    leakage_signal = abs(leakage)
    denominator = target_signal + leakage_signal
    if denominator <= FORM_QUALITY_EPSILON:
        # No target transmission receives zero quality credit rather than
        # becoming an unevaluable candidate.
        return 0.0
    return _number(target_signal / denominator)


def form_quality_summary(
    form_metrics: Mapping[str, Any],
    *,
    stability_minimum: float = VIRTUAL_FORM_ICC_DEFAULT_MINIMUM,
) -> dict[str, Any]:
    """Extract the predeclared candidate-form utility and its ICC gate.

    Target recovery and construct selectivity are combined with a geometric
    mean.  ICC is deliberately a feasibility gate rather than an optimization
    component because virtual retest values are commonly ceiling-limited.
    """

    if (
        isinstance(stability_minimum, bool)
        or not isinstance(stability_minimum, (int, float))
        or not np.isfinite(float(stability_minimum))
        or not 0.0 <= float(stability_minimum) <= 1.0
    ):
        raise ValueError("虚拟整卷 ICC 门槛必须是 0 到 1 之间的数值")

    reliability = form_metrics.get("reliability") or {}
    validity = form_metrics.get("validity") or {}
    recovery = validity.get("target_recovery") or {}
    selectivity = validity.get("construct_selectivity") or {}
    isolation = validity.get("construct_isolation") or {}

    recovery_raw = _number(recovery.get("cross_validated_r2"))
    selectivity_value = _number(selectivity.get("value"))
    if selectivity_value is None:
        target_effect = isolation.get("target_sensitivity") or {}
        selectivity_value = _construct_selectivity_value(
            target_effect.get("standardized_effect"),
            isolation.get("maximum_absolute_non_target_leakage"),
        )
    icc = _number(reliability.get("virtual_test_retest_icc"))

    recovery_component = (
        min(1.0, max(0.0, recovery_raw))
        if recovery_raw is not None
        else None
    )
    selectivity_component = (
        min(1.0, max(0.0, selectivity_value))
        if selectivity_value is not None
        else None
    )
    candidate_quality = (
        _number(float(np.sqrt(recovery_component * selectivity_component)))
        if recovery_component is not None and selectivity_component is not None
        else None
    )
    stability_passed = (
        icc is not None and icc >= float(stability_minimum)
    )
    complete = candidate_quality is not None and icc is not None
    return {
        "status": "complete" if complete else "unavailable",
        "candidate_form_quality": candidate_quality,
        "target_recovery_raw": recovery_raw,
        "target_recovery_component": recovery_component,
        "construct_selectivity": selectivity_component,
        "aggregation": "geometric_mean",
        "formula": "sqrt(clipped_target_recovery_r2 * construct_selectivity)",
        "stability_gate": {
            "metric": "virtual_test_retest_icc",
            "minimum": float(stability_minimum),
            "observed": icc,
            "passed": stability_passed,
        },
        "eligible_for_best_so_far": bool(complete and stability_passed),
        "interpretation": (
            "虚拟开发期候选整卷质量；不是人类样本信度或效度。"
        ),
    }


def assess_form_plateau(
    history: Sequence[Mapping[str, Any]],
    *,
    patience: int = PLATEAU_DEFAULT_PATIENCE,
    min_delta: float = PLATEAU_DEFAULT_MIN_DELTA,
) -> dict[str, Any]:
    """Detect a plateau in the retained best-so-far whole-form quality.

    A complete, blueprint-valid round enters the comparison only after its ICC
    stability gate passes.  Its scalar quality is the geometric mean of target
    recovery and construct selectivity.  A candidate replaces the incumbent
    only when it exceeds the retained best by ``min_delta``.
    """

    if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
        raise ValueError("平台期 patience 必须是正整数")
    if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)) or min_delta < 0:
        raise ValueError("平台期 min_delta 必须是非负数")
    trajectory: list[dict[str, Any]] = []
    best_round: int | None = None
    best_quality: float | None = None
    best_summary: dict[str, Any] | None = None
    non_improving = 0
    reached_round: int | None = None
    usable_rounds = 0
    current_round: int | None = None
    current_summary: dict[str, Any] | None = None

    ordered_history = sorted(
        (entry for entry in history if isinstance(entry, Mapping)),
        key=lambda entry: int(entry.get("analysis_round") or 0),
    )
    for entry in ordered_history:
        round_number = int(entry.get("analysis_round") or 0)
        current_round = round_number
        summary = form_quality_summary(entry.get("form_metrics") or {})
        current_summary = summary
        complete_form = entry.get("form_status") in (None, "complete")
        quality = _number(summary.get("candidate_form_quality"))
        eligible = bool(
            complete_form
            and summary.get("eligible_for_best_so_far")
            and quality is not None
        )
        accepted = False
        if eligible:
            usable_rounds += 1
            if best_quality is None or quality > best_quality + float(min_delta):
                best_quality = quality
                best_round = round_number
                best_summary = summary
                non_improving = 0
                accepted = True
            else:
                non_improving += 1
                if non_improving >= patience and reached_round is None:
                    reached_round = round_number
        trajectory.append(
            {
                "analysis_round": round_number,
                "candidate_form_quality": quality,
                "best_so_far_form_quality": best_quality,
                "accepted_as_best": accepted,
                "eligible_for_best_so_far": eligible,
                "stability_gate": dict(summary.get("stability_gate") or {}),
                "non_improving_rounds": non_improving,
            }
        )
    base = {
        "status": "insufficient_data",
        "reached": False,
        "patience": patience,
        "min_delta": float(min_delta),
        "usable_rounds": usable_rounds,
        "non_improving_rounds": non_improving,
        "best_round": best_round,
        "current_round": current_round,
        "best_form_quality": best_quality,
        "current_candidate_form_quality": (
            current_summary.get("candidate_form_quality")
            if current_summary is not None
            else None
        ),
        "best_metrics": best_summary,
        "current_metrics": current_summary,
        "metric_names": [
            "candidate_form_quality",
            "best_so_far_form_quality",
        ],
        "stability_gate_metric": "virtual_test_retest_icc",
        "trajectory": trajectory,
    }
    if not usable_rounds:
        base["reason"] = "尚无通过 ICC 稳定性门槛的完整整卷轮次"
        return base

    base.update(
        {
            "status": "reached" if reached_round is not None else "monitoring",
            "reached": reached_round is not None,
            "non_improving_rounds": non_improving,
            "best_round": best_round,
            "plateau_round": reached_round,
            "reason": (
                f"连续 {non_improving} 轮候选整卷未使历史最优质量提高至少 {float(min_delta):.3f}"
                if reached_round is not None
                else "继续观察后续整卷轮次"
            ),
        }
    )
    return base


def build_provisional_form_metrics(
    state: Mapping[str, Any],
    selected_item_ids: Sequence[str],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one complete test as a virtual response-transmission system."""

    item_ids = list(dict.fromkeys(str(item_id) for item_id in selected_item_ids))
    if not item_ids:
        return {
            "status": "unavailable",
            "reason": "临时测验没有可计算的题目",
            "item_count": 0,
            "selected_item_ids": [],
        }
    metric_context = (
        context
        if isinstance(context, Mapping)
        else prepare_provisional_form_metric_context(state)
    )
    responses = metric_context.get("responses")
    condition_matrices = metric_context.get("condition_item_scores")
    condition_active = metric_context.get("condition_active_scores")
    condition_metadata = metric_context.get("condition_metadata")
    if (
        not isinstance(responses, pd.DataFrame)
        or not isinstance(condition_matrices, Mapping)
        or not isinstance(condition_active, Mapping)
        or "target" not in condition_matrices
        or "target" not in condition_active
    ):
        return {
            "status": "unavailable",
            "reason": "当前轮次缺少完整的匹配条件逐题作答数据",
            "item_count": len(item_ids),
            "selected_item_ids": item_ids,
        }
    target_scores = condition_matrices["target"]
    target_active = condition_active["target"]
    if not isinstance(target_scores, pd.DataFrame) or not isinstance(
        target_active, pd.Series
    ):
        return {
            "status": "unavailable",
            "reason": "target 条件数据结构无效",
            "item_count": len(item_ids),
            "selected_item_ids": item_ids,
        }
    missing = [
        item_id for item_id in item_ids if item_id not in target_scores.columns
    ]
    if missing:
        return {
            "status": "unavailable",
            "reason": "逐题作答数据缺少临时测验题目：" + ", ".join(missing),
            "item_count": len(item_ids),
            "selected_item_ids": item_ids,
        }
    target_matrix = target_scores[item_ids].dropna(axis=0, how="any")
    form_score = target_matrix.mean(axis=1)

    # 1) Virtual whole-form reliability: repeat the same target personas and
    # calculate absolute-agreement ICC on complete-form scores.
    administration_scores = {"1": form_score}
    target_retests = metric_context.get("target_retest_scores")
    if isinstance(target_retests, Mapping):
        for administration_id, matrix in target_retests.items():
            if not isinstance(matrix, pd.DataFrame) or any(
                item_id not in matrix.columns for item_id in item_ids
            ):
                continue
            administration_scores[str(administration_id)] = matrix[
                item_ids
            ].dropna(axis=0, how="any").mean(axis=1)
    stability_frame = pd.concat(administration_scores, axis=1, join="inner")
    stability_icc = _icc_absolute_agreement_single(stability_frame)
    stability = {
        "status": "complete" if stability_icc is not None else "unavailable",
        "virtual_test_retest_icc": stability_icc,
        "method": "ICC(A,1)_absolute_agreement_single_measure",
        "sample_size": int(len(stability_frame)),
        "administration_count": int(stability_frame.shape[1]),
        "administration_ids": list(stability_frame.columns),
        "interpretation": (
            "同一 target 虚拟人格重复完成整套测验时的总分绝对一致性；"
            "不是人类样本信度。"
        ),
    }
    if stability_icc is None:
        stability["reason"] = "缺少至少两次完整 target 整卷施测或总分无变异"

    # 2) Target recovery: held-out prediction from the complete response
    # pattern, not an average of item-level target correlations.
    recovery = _cross_validated_target_recovery(target_matrix, target_active)
    recovery["interpretation"] = (
        "使用整套题目的完整作答模式，在留出虚拟被试上恢复预设目标构念；"
        "不是人类汇聚效度或校标效度。"
    )

    # 3) Construct isolation: compare target sensitivity with the largest
    # high-low score effect produced by any non-target condition's own form.
    target_sd = _number(float(form_score.std(ddof=1)))
    target_effect = _extreme_group_effect(
        form_score,
        target_active,
        denominator_sd=target_sd or 0.0,
    )
    leakage_groups: list[dict[str, Any]] = []
    for condition_id, matrix in condition_matrices.items():
        condition_id = str(condition_id)
        if condition_id == "target" or not isinstance(matrix, pd.DataFrame):
            continue
        if any(item_id not in matrix.columns for item_id in item_ids):
            continue
        active = condition_active.get(condition_id)
        if not isinstance(active, pd.Series):
            continue
        effect = _extreme_group_effect(
            matrix[item_ids].dropna(axis=0, how="any").mean(axis=1),
            active,
            denominator_sd=target_sd or 0.0,
        )
        metadata = (
            condition_metadata.get(condition_id, {})
            if isinstance(condition_metadata, Mapping)
            else {}
        )
        leakage_groups.append(
            {
                "condition_id": condition_id,
                "arm_id": metadata.get("arm_id"),
                "group_id": metadata.get("group_id"),
                **effect,
            }
        )
    estimable_leakage = [
        row
        for row in leakage_groups
        if isinstance(row.get("standardized_effect"), (int, float))
        and not isinstance(row.get("standardized_effect"), bool)
    ]
    maximum_leakage = (
        max(abs(float(row["standardized_effect"])) for row in estimable_leakage)
        if estimable_leakage
        else None
    )
    target_sensitivity = target_effect.get("standardized_effect")
    isolation_value = (
        float(target_sensitivity) - float(maximum_leakage)
        if isinstance(target_sensitivity, (int, float))
        and maximum_leakage is not None
        else None
    )
    selectivity_value = _construct_selectivity_value(
        target_sensitivity,
        maximum_leakage,
    )
    construct_isolation = {
        "status": "complete" if isolation_value is not None else "unavailable",
        "value": _number(isolation_value),
        "target_sensitivity": target_effect,
        "maximum_absolute_non_target_leakage": _number(maximum_leakage),
        "non_target_groups": leakage_groups,
        "effect_definition": (
            "upper-third minus lower-third form-score mean, divided by the "
            "target-arm form-score SD"
        ),
        "interpretation": (
            "目标构念高低变化引起的整卷效应减去最大非目标构念泄漏；"
            "不是人类区分效度。"
        ),
    }
    construct_selectivity = {
        "status": "complete" if selectivity_value is not None else "unavailable",
        "value": _number(selectivity_value),
        "target_sensitivity": _number(target_sensitivity),
        "maximum_absolute_non_target_leakage": _number(maximum_leakage),
        "target_direction_passed": bool(
            isinstance(target_sensitivity, (int, float))
            and not isinstance(target_sensitivity, bool)
            and float(target_sensitivity) > 0.0
        ),
        "formula": "max(0,T) / (max(0,T) + abs(L))",
        "interpretation": (
            "目标构念信号占目标信号与最大非目标泄漏总量的比例；"
            "仅用于虚拟开发期构念特异性评价。"
        ),
    }

    # Human-style quantities remain descriptive diagnostics only.  They are
    # retained for backward-compatible reports but are not iteration targets.
    alpha = _cronbach_alpha(target_matrix)
    target_aligned = pd.concat(
        [form_score.rename("form_score"), target_active.rename("active_score")],
        axis=1,
        join="inner",
    ).dropna()
    target_rho = _spearman(
        target_aligned["form_score"].tolist(),
        target_aligned["active_score"].tolist(),
    )
    reference_result = metric_context.get("reference_result")
    if not isinstance(reference_result, Mapping):
        reference_result = {"status": "unavailable"}
    reference_scores = reference_result.get("scores")
    neo_rho = None
    if isinstance(reference_scores, pd.DataFrame):
        aligned_reference = pd.concat(
            [form_score.rename("form_score"), reference_scores],
            axis=1,
            join="inner",
        )
        if "neo_ffi_target_score" in aligned_reference:
            neo = aligned_reference[
                ["form_score", "neo_ffi_target_score"]
            ].dropna()
            neo_rho = _spearman(
                neo["form_score"].tolist(), neo["neo_ffi_target_score"].tolist()
            )
    legacy_reference_rho = neo_rho

    complete = all(
        value is not None
        for value in (
            stability_icc,
            recovery.get("cross_validated_r2"),
            selectivity_value,
        )
    )
    result = {
        "status": "complete" if complete else "partial",
        "selected_item_ids": item_ids,
        "item_count": len(item_ids),
        "sample_size": int(len(target_matrix)),
        "metric_framework": "virtual_form_response_transmission_v2",
        "reliability": {
            **stability,
            "cronbach_alpha": alpha,
            "cronbach_alpha_role": "descriptive_only_not_iteration_target",
        },
        "validity": {
            "target_recovery": recovery,
            "construct_selectivity": construct_selectivity,
            "construct_isolation": construct_isolation,
            "legacy_virtual_diagnostics": {
                "target_total_score_spearman": target_rho,
                "neo_ffi_rho": neo_rho,
                "combined_reference_rho": legacy_reference_rho,
                "reference_details": reference_result.get("details") or {},
                "filtering_authority": False,
            },
            "criterion_validity": {
                "status": "not_available",
                "value": None,
                "reason": (
                    "active_score、Neo-FFI与Mussel均由同一虚拟人格生成过程产生，"
                    "不能作为独立外部校标。"
                ),
            },
        },
        "iteration_objectives": [
            "optimization.candidate_form_quality",
        ],
        "iteration_components": [
            "validity.target_recovery.cross_validated_r2",
            "validity.construct_selectivity.value",
        ],
        "iteration_constraints": [
            "optimization.stability_gate",
        ],
        "interpretation": (
            "目标恢复度与构念选择性共同评价候选整卷质量，虚拟重测ICC仅作为稳定性门槛；"
            "仅用于开发期迭代，不能替代真人信效度验证。"
        ),
    }
    result["optimization"] = form_quality_summary(result)
    return result


def batch_provisional_form_quality(
    context: Mapping[str, Any] | None,
    item_combinations: Sequence[Sequence[str]],
) -> dict[str, np.ndarray] | None:
    """Fast first-stage ranking proxies for many complete forms.

    The final shortlisted forms are always rescored with the full ICC,
    cross-validated response-pattern recovery, and construct-isolation
    formulas.  This vectorized pass only narrows a potentially large search.
    """

    if not isinstance(context, Mapping) or not item_combinations:
        return None
    target_scores = context.get("target_scores")
    target_active = context.get("target_active")
    condition_matrices = context.get("condition_item_scores")
    condition_active = context.get("condition_active_scores")
    target_retests = context.get("target_retest_scores")
    if (
        not isinstance(target_scores, pd.DataFrame)
        or not isinstance(target_active, pd.Series)
        or not isinstance(condition_matrices, Mapping)
        or not isinstance(condition_active, Mapping)
        or not isinstance(target_retests, Mapping)
        or not target_retests
    ):
        return None
    combinations = [list(map(str, row)) for row in item_combinations]
    item_count = len(combinations[0])
    if item_count < 2 or any(
        len(row) != item_count or len(set(row)) != item_count
        for row in combinations
    ):
        return None
    candidate_ids = list(
        dict.fromkeys(item_id for row in combinations for item_id in row)
    )
    if any(item_id not in target_scores.columns for item_id in candidate_ids):
        return None
    item_scores = target_scores[candidate_ids].to_numpy(dtype=float)
    if item_scores.ndim != 2 or not np.isfinite(item_scores).all():
        return None
    id_to_index = {
        item_id: index for index, item_id in enumerate(candidate_ids)
    }
    selection = np.zeros(
        (len(combinations), len(candidate_ids)), dtype=float
    )
    for row_index, row in enumerate(combinations):
        selection[
            row_index, [id_to_index[item_id] for item_id in row]
        ] = 1.0
    form_scores = selection @ item_scores.T / float(item_count)

    def rowwise_spearman(
        values: np.ndarray, criterion: pd.Series
    ) -> np.ndarray | None:
        aligned = criterion.reindex(target_scores.index).to_numpy(dtype=float)
        if aligned.ndim != 1 or not np.isfinite(aligned).all():
            return None
        ranked_values = stats.rankdata(values, axis=1, method="average")
        ranked_criterion = stats.rankdata(aligned, method="average")
        centered_values = ranked_values - ranked_values.mean(
            axis=1, keepdims=True
        )
        centered_criterion = ranked_criterion - ranked_criterion.mean()
        numerator = centered_values @ centered_criterion
        denominator = np.sqrt(
            (centered_values * centered_values).sum(axis=1)
            * float((centered_criterion * centered_criterion).sum())
        )
        return np.divide(
            numerator,
            denominator,
            out=np.full(len(values), np.nan, dtype=float),
            where=denominator > 0,
        )

    target_rho = rowwise_spearman(form_scores, target_active)
    if target_rho is None:
        return None
    # Signed squared correlation is only a screening proxy.  Full candidates
    # use held-out ridge prediction from every selected item response.
    recovery_proxy = target_rho * np.abs(target_rho)

    target_sd = form_scores.std(axis=1, ddof=1)
    target_sd = np.where(target_sd > 0, target_sd, np.nan)

    def contrast(
        values: np.ndarray, active: pd.Series
    ) -> np.ndarray | None:
        aligned = active.reindex(target_scores.index).to_numpy(dtype=float)
        if not np.isfinite(aligned).all():
            return None
        low_cut = float(np.quantile(aligned, FORM_EFFECT_EXTREME_FRACTION))
        high_cut = float(
            np.quantile(aligned, 1.0 - FORM_EFFECT_EXTREME_FRACTION)
        )
        low = aligned <= low_cut
        high = aligned >= high_cut
        if int(low.sum()) < 2 or int(high.sum()) < 2:
            return None
        return (
            values[:, high].mean(axis=1)
            - values[:, low].mean(axis=1)
        ) / target_sd

    target_effect = contrast(form_scores, target_active)
    if target_effect is None:
        return None
    leakage_effects: list[np.ndarray] = []
    for condition_id, matrix in condition_matrices.items():
        if str(condition_id) == "target" or not isinstance(matrix, pd.DataFrame):
            continue
        if any(item_id not in matrix.columns for item_id in candidate_ids):
            return None
        active = condition_active.get(condition_id)
        if not isinstance(active, pd.Series):
            return None
        condition_items = matrix.reindex(target_scores.index)[
            candidate_ids
        ].to_numpy(dtype=float)
        if not np.isfinite(condition_items).all():
            return None
        condition_forms = selection @ condition_items.T / float(item_count)
        effect = contrast(condition_forms, active)
        if effect is None:
            return None
        leakage_effects.append(np.abs(effect))
    if not leakage_effects:
        return None
    maximum_leakage = np.max(np.vstack(leakage_effects), axis=0)
    construct_isolation = target_effect - maximum_leakage
    positive_target = np.maximum(target_effect, 0.0)
    selectivity_denominator = positive_target + maximum_leakage
    construct_selectivity = np.divide(
        positive_target,
        selectivity_denominator,
        out=np.zeros(len(positive_target), dtype=float),
        where=selectivity_denominator > FORM_QUALITY_EPSILON,
    )
    candidate_form_quality_proxy = np.sqrt(
        np.clip(recovery_proxy, 0.0, 1.0)
        * np.clip(construct_selectivity, 0.0, 1.0)
    )

    def concordance(
        primary: np.ndarray, repeated: np.ndarray
    ) -> np.ndarray:
        primary_mean = primary.mean(axis=1)
        repeated_mean = repeated.mean(axis=1)
        centered_primary = primary - primary_mean[:, None]
        centered_repeated = repeated - repeated_mean[:, None]
        covariance = (
            centered_primary * centered_repeated
        ).mean(axis=1)
        denominator = (
            np.square(centered_primary).mean(axis=1)
            + np.square(centered_repeated).mean(axis=1)
            + np.square(primary_mean - repeated_mean)
        )
        return np.divide(
            2.0 * covariance,
            denominator,
            out=np.full(len(primary), np.nan, dtype=float),
            where=denominator > 0,
        )

    stability_proxies: list[np.ndarray] = []
    for matrix in target_retests.values():
        if not isinstance(matrix, pd.DataFrame) or any(
            item_id not in matrix.columns for item_id in candidate_ids
        ):
            return None
        repeated_items = matrix.reindex(target_scores.index)[
            candidate_ids
        ].to_numpy(dtype=float)
        if not np.isfinite(repeated_items).all():
            return None
        repeated_forms = selection @ repeated_items.T / float(item_count)
        stability_proxies.append(concordance(form_scores, repeated_forms))
    stability_proxy = np.min(np.vstack(stability_proxies), axis=0)
    values = {
        "target_recovery_proxy": recovery_proxy,
        "construct_selectivity": construct_selectivity,
        "candidate_form_quality_proxy": candidate_form_quality_proxy,
        "construct_isolation": construct_isolation,
        "stability_proxy": stability_proxy,
    }
    if any(not np.isfinite(array).all() for array in values.values()):
        return None
    return values
