"""Deterministic scoring and psychometric analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from sjt_system.authoring.construct_registry import (
    resolve_neo_ffi_criterion,
)
from sjt_system.evaluation.respondents import (
    PERSONA_MODE_SCORE_PROFILE,
    PERSONA_MODE_SUMMARY_PLUS_ITEMS,
    score_spec_is_target_related,
    MATCHED_CONDITION_IDS,
    MATCHED_CONDITION_SCHEMA_VERSION,
    flatten_matched_condition_groups,
)
from sjt_system.state import PSJTState, create_initial_state
from sjt_system.runtime.trace import utc_timestamp


PSYCHOMETRIC_FORMULA_VERSION = "sjt-matched-facet-virtual-screening-v10"
MEASUREMENT_EVALUATION_VERSION = "sjt-evaluation-v11"
OPTION_CHOICE_DIAGNOSTICS_VERSION = "matched-condition-option-choice-v3"
MINIMUM_OPTION_DIAGNOSTIC_GROUP_N = 10
DIFFICULTY_LOWER_BOUND = 0.20
DIFFICULTY_UPPER_BOUND = 0.80
MINIMUM_OPTION_RATE = 0.05
CITC_MINIMUM_ACCEPTABLE = 0.30
CITC_REVISION_THRESHOLD = 0.20
CITC_STRONG = 0.50
TARGET_RHO_THRESHOLD = 0.30
SAME_DOMAIN_VTS_THRESHOLD = 0.10
CROSS_DOMAIN_VTS_THRESHOLD = 0.20


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} 必须包含 JSON 对象")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name} 第 {line_number} 行不是有效 JSON"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path.name} 第 {line_number} 行必须是 JSON 对象"
                )
            records.append(record)
    if not records:
        raise ValueError(f"{path.name} 没有作答记录")
    return records


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value.resolve())
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    temporary_path.replace(path)


def _write_jsonl_atomic(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        "".join(
            json.dumps(dict(record), ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def standardized_item_difficulty(
    scores: Sequence[float],
    theoretical_minimum: float,
    theoretical_maximum: float,
) -> float:
    """Return the polytomous item mean normalized to its theoretical range."""

    if theoretical_maximum <= theoretical_minimum:
        raise ValueError("理论最高分必须大于理论最低分")
    values = np.asarray(scores, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("难度计算需要非空且有限的题目得分")
    return float(
        (values.mean() - theoretical_minimum)
        / (theoretical_maximum - theoretical_minimum)
    )


def cronbach_alpha(matrix: np.ndarray) -> float | None:
    """Compute raw Cronbach alpha with sample variances (ddof=1)."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("Cronbach alpha 输入必须是二维矩阵")
    sample_count, item_count = values.shape
    if sample_count < 2 or item_count < 2:
        return None
    if not np.isfinite(values).all():
        raise ValueError("Cronbach alpha 输入不能包含缺失值或无穷值")
    item_variances = values.var(axis=0, ddof=1)
    total_variance = values.sum(axis=1).var(ddof=1)
    if total_variance <= 0:
        return None
    return float(
        item_count
        / (item_count - 1)
        * (1 - item_variances.sum() / total_variance)
    )


def _pearson(
    left: Sequence[float],
    right: Sequence[float],
) -> dict[str, Any]:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"r": None, "p_value": None, "n": int(x.size)}
    result = stats.pearsonr(x, y)
    return {
        "r": float(result.statistic),
        "p_value": float(result.pvalue),
        "n": int(x.size),
    }


def _spearman(
    left: Sequence[float],
    right: Sequence[float],
) -> dict[str, Any]:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"rho": None, "p_value": None, "n": int(x.size)}
    result = stats.spearmanr(x, y)
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "n": int(x.size),
    }


def _conditional_spearman(
    left: Sequence[float],
    right: Sequence[float],
    tier_ids: Sequence[Any],
) -> dict[str, Any]:
    """Spearman association after residualizing average ranks on tier dummies."""

    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    tiers = np.asarray(tier_ids, dtype=object)
    if x.size != y.size or x.size != tiers.size:
        raise ValueError("条件Spearman的变量与tier_id长度不一致")
    valid = np.isfinite(x) & np.isfinite(y) & pd.notna(tiers)
    x = x[valid]
    y = y[valid]
    tiers = tiers[valid].astype(str)
    unique_tiers = sorted(set(tiers.tolist()))
    base = {
        "conditioning_variable": "tier_id",
        "method": "rank_residualization",
        "rank_method": "average",
        "tier_count": len(unique_tiers),
    }
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0 or not unique_tiers:
        return {"rho": None, "p_value": None, "n": int(x.size), **base}
    if len(unique_tiers) == 1:
        return {**_spearman(x, y), **base}

    design = np.column_stack(
        [
            np.ones(x.size, dtype=float),
            *(tiers == tier_id for tier_id in unique_tiers[1:]),
        ]
    ).astype(float)
    ranked_x = stats.rankdata(x, method="average")
    ranked_y = stats.rankdata(y, method="average")
    residual_x = ranked_x - design @ np.linalg.lstsq(design, ranked_x, rcond=None)[0]
    residual_y = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    if np.allclose(residual_x, 0.0, atol=1e-12) or np.allclose(
        residual_y, 0.0, atol=1e-12
    ):
        return {"rho": None, "p_value": None, "n": int(x.size), **base}
    result = stats.pearsonr(residual_x, residual_y)
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "n": int(x.size),
        **base,
    }


def _conditional_score_band_assignments(
    values: pd.Series,
    tier_ids: pd.Series,
) -> tuple[pd.Series, dict[str, Any]]:
    """Split tier-controlled average-rank residuals into diagnostic tertiles."""

    if not values.index.equals(tier_ids.index):
        tier_ids = tier_ids.reindex(values.index)
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & tier_ids.notna()
    x = numeric.loc[valid].to_numpy(dtype=float)
    tiers = tier_ids.loc[valid].astype(str).to_numpy(dtype=object)
    unique_tiers = sorted(set(tiers.tolist()))
    assignments = pd.Series(index=values.index, dtype="object")
    base = {
        "conditioning_variable": "tier_id",
        "method": "average_rank_residual_tertiles",
        "rank_method": "average",
        "tier_count": len(unique_tiers),
        "minimum_group_n": MINIMUM_OPTION_DIAGNOSTIC_GROUP_N,
        "filtering_authority": False,
    }
    if x.size < 3 or np.ptp(x) == 0 or not unique_tiers:
        return assignments, {
            **base,
            "estimable": False,
            "reason": "构念分数缺少可用于条件分组的方差",
            "group_sizes": {"low": 0, "medium": 0, "high": 0},
        }

    ranked = stats.rankdata(x, method="average")
    if len(unique_tiers) == 1:
        residuals = ranked - float(np.mean(ranked))
    else:
        design = np.column_stack(
            [
                np.ones(x.size, dtype=float),
                *(tiers == tier_id for tier_id in unique_tiers[1:]),
            ]
        ).astype(float)
        residuals = ranked - design @ np.linalg.lstsq(
            design, ranked, rcond=None
        )[0]
    if np.allclose(residuals, 0.0, atol=1e-12):
        return assignments, {
            **base,
            "estimable": False,
            "reason": "控制tier_id后的构念秩残差无方差",
            "group_sizes": {"low": 0, "medium": 0, "high": 0},
        }

    residual_ranks = stats.rankdata(residuals, method="average")
    percentiles = residual_ranks / float(x.size)
    labels = np.where(
        percentiles <= (1.0 / 3.0),
        "low",
        np.where(percentiles <= (2.0 / 3.0), "medium", "high"),
    )
    assignments.loc[valid] = labels
    group_sizes = {
        label: int(np.sum(labels == label))
        for label in ("low", "medium", "high")
    }
    estimable = all(
        group_sizes[label] >= MINIMUM_OPTION_DIAGNOSTIC_GROUP_N
        for label in ("low", "medium", "high")
    )
    return assignments, {
        **base,
        "estimable": estimable,
        "reason": (
            None
            if estimable
            else "至少一个条件分组少于10人；频率仅展示，不得用于返修推断"
        ),
        "group_sizes": group_sizes,
    }


def _validate_analysis_identity(
    state: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    if manifest.get("status") != "completed":
        raise ValueError("只有 completed 的虚拟作答数据可以进入分析")
    expected = {
        "run_id": state.get("run_id"),
        "item_bank_id": state.get("item_bank_id"),
        "item_bank_version": state.get("item_bank_version"),
        "item_bank_fingerprint": state.get("item_bank_fingerprint"),
    }
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            raise ValueError(
                f"虚拟作答 {field} 与当前冻结题库不一致："
                f"{manifest.get(field)!r} != {expected_value!r}"
            )
    if state.get("virtual_response_item_bank_id") != state.get(
        "item_bank_id"
    ):
        raise ValueError("State 中虚拟作答绑定的 item_bank_id 已失效")
    if state.get("virtual_response_item_bank_version") != state.get(
        "item_bank_version"
    ):
        raise ValueError("State 中虚拟作答绑定的题库版本已失效")


def _resolve_response_paths(
    state: Mapping[str, Any],
) -> tuple[Path, Path, Path, dict[str, Any]]:
    reference = state.get("virtual_response_data_ref")
    if not isinstance(reference, str) or not reference:
        raise ValueError("心理测量分析前缺少 virtual_response_data_ref")
    manifest_path = Path(reference).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"虚拟作答 manifest 不存在：{manifest_path}")
    manifest = _read_json(manifest_path)
    _validate_analysis_identity(state, manifest)
    sjt_path = manifest_path.parent / "sjt_responses.jsonl"
    neo_path = manifest_path.parent / "neo_ffi_responses.jsonl"
    if not sjt_path.is_file():
        raise ValueError("虚拟作答目录缺少 SJT JSONL 文件")
    if manifest.get("schema_version") not in {2, 3, MATCHED_CONDITION_SCHEMA_VERSION} and not neo_path.is_file():
        raise ValueError("旧版虚拟作答目录缺少 Neo-FFI JSONL 文件")
    return manifest_path, sjt_path, neo_path, manifest


def _resolve_analysis_criterion(
    state: Mapping[str, Any],
    response_manifest: Mapping[str, Any],
) -> dict[str, str]:
    profile = state.get("construct_profile")
    if not isinstance(profile, Mapping):
        blueprint = state.get("blueprint")
        if isinstance(blueprint, Mapping):
            candidate = blueprint.get("construct_profile_snapshot")
            if isinstance(candidate, Mapping):
                profile = candidate
    manifest_domain = response_manifest.get("criterion_domain_id")
    manifest_dimension = response_manifest.get(
        "criterion_neo_ffi_dimension"
    )
    manifest_criterion: dict[str, str] | None = None
    if manifest_domain is not None or manifest_dimension is not None:
        if not isinstance(manifest_domain, str) or not isinstance(
            manifest_dimension,
            str,
        ):
            raise ValueError(
                "response manifest 的 criterion domain/Neo dimension "
                "必须同时存在"
            )
        manifest_criterion = resolve_neo_ffi_criterion(
            construct_profile={"domain_id": manifest_domain}
        )
        if (
            manifest_criterion["neo_ffi_dimension"]
            != manifest_dimension
        ):
            raise ValueError(
                "response manifest 的 criterion domain 与 Neo dimension "
                "映射不一致"
            )

    state_criterion: dict[str, str] | None = None
    if isinstance(profile, Mapping):
        state_criterion = resolve_neo_ffi_criterion(
            construct_profile=profile,
        )
    if (
        manifest_criterion is not None
        and state_criterion is not None
        and manifest_criterion["domain_id"]
        != state_criterion["domain_id"]
    ):
        raise ValueError(
            "虚拟作答 manifest 的目标构念与当前 State 不一致"
        )
    criterion = state_criterion or manifest_criterion
    if criterion is None:
        raise ValueError(
            "心理测量分析缺少目标构念到 Neo-FFI 的确定性映射"
        )
    return criterion


def _prepare_item_contract(
    state: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    frozen = state.get("frozen_item_bank")
    if not isinstance(frozen, list) or not frozen:
        raise ValueError("心理测量分析前缺少冻结题库")
    item_order: list[str] = []
    items: dict[str, dict[str, Any]] = {}
    for raw_item in frozen:
        if not isinstance(raw_item, Mapping):
            raise ValueError("冻结题库包含无效题目")
        item = dict(raw_item)
        item_id = item.get("item_id")
        scoring_key = item.get("scoring_key")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("冻结题库包含缺少 item_id 的题目")
        if item_id in items:
            raise ValueError(f"冻结题库包含重复题目：{item_id}")
        if not isinstance(scoring_key, Mapping) or len(scoring_key) < 2:
            raise ValueError(f"题目 {item_id} 缺少有效 scoring_key")
        if not isinstance(item.get("target_dimension_id"), str) or not item[
            "target_dimension_id"
        ].strip():
            raise ValueError(
                f"题目 {item_id} 缺少有效 target_dimension_id"
            )
        scores = list(scoring_key.values())
        if any(
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            for score in scores
        ):
            raise ValueError(f"题目 {item_id} 的 scoring_key 包含无效分数")
        item_order.append(item_id)
        items[item_id] = item
    return item_order, items


def _score_sjt(
    records: Sequence[Mapping[str, Any]],
    *,
    item_order: list[str],
    items: Mapping[str, Mapping[str, Any]],
    expected_respondent_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        respondent_id = record.get("respondent_id")
        item_id = record.get("item_id")
        option_id = record.get("selected_option_id")
        if not isinstance(respondent_id, str) or not respondent_id:
            raise ValueError("SJT 作答包含无效 respondent_id")
        if item_id not in items:
            raise ValueError(f"SJT 作答引用未知题目：{item_id!r}")
        key = (respondent_id, str(item_id))
        if key in seen:
            raise ValueError(f"SJT 作答包含重复记录：{key}")
        seen.add(key)
        item = items[str(item_id)]
        expected_version = item.get("version")
        if record.get("item_version") != expected_version:
            raise ValueError(
                f"题目 {item_id} 的作答版本与冻结题库不一致"
            )
        scoring_key = item["scoring_key"]
        if option_id not in scoring_key:
            raise ValueError(
                f"题目 {item_id} 选择了 scoring_key 中不存在的选项"
            )
        rows.append(
            {
                "respondent_id": respondent_id,
                "item_id": item_id,
                "item_version": expected_version,
                "dimension_id": item.get("target_dimension_id"),
                "context_category": item.get("context_category"),
                "selected_option_id": option_id,
                "score": float(scoring_key[option_id]),
            }
        )

    long_frame = pd.DataFrame(rows)
    respondent_ids = sorted(long_frame["respondent_id"].unique())
    if len(respondent_ids) != expected_respondent_count:
        raise ValueError(
            "SJT 实际被试数与 manifest 不一致："
            f"{len(respondent_ids)} != {expected_respondent_count}"
        )
    expected_records = expected_respondent_count * len(item_order)
    if len(long_frame) != expected_records:
        raise ValueError(
            f"SJT 作答不完整：应有 {expected_records} 条，"
            f"实际 {len(long_frame)} 条"
        )
    wide = long_frame.pivot(
        index="respondent_id",
        columns="item_id",
        values="score",
    ).reindex(index=respondent_ids, columns=item_order)
    if wide.isna().any().any():
        raise ValueError("SJT 被试×题目矩阵存在缺失作答")
    return long_frame, wide


def _score_tiered_sjt(
    records: Sequence[Mapping[str, Any]],
    *,
    item_order: list[str],
    items: Mapping[str, Mapping[str, Any]],
    expected_respondent_count: int,
    expected_tier_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Score one response per respondent/item and split matrices by tier."""

    tier_ids = [str(value) for value in expected_tier_ids]
    if not tier_ids or len(set(tier_ids)) != len(tier_ids):
        raise ValueError("作答 manifest 缺少有效分数档")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    respondent_tiers: dict[str, str] = {}
    for record in records:
        respondent_id = record.get("respondent_id")
        item_id = record.get("item_id")
        option_id = record.get("selected_option_id")
        tier_id = record.get("tier_id")
        if not isinstance(respondent_id, str) or not respondent_id:
            raise ValueError("SJT 作答包含无效 respondent_id")
        if item_id not in items:
            raise ValueError(f"SJT 作答引用未知题目：{item_id!r}")
        if tier_id not in tier_ids:
            raise ValueError("SJT 作答缺少有效 tier_id")
        if respondent_id in respondent_tiers and respondent_tiers[respondent_id] != tier_id:
            raise ValueError("同一虚拟被试不能跨越多个分数档")
        respondent_tiers[respondent_id] = str(tier_id)
        key = (respondent_id, str(item_id))
        if key in seen:
            raise ValueError(f"SJT 作答包含重复记录：{key}")
        seen.add(key)
        item = items[str(item_id)]
        expected_version = item.get("version")
        if record.get("item_version") != expected_version:
            raise ValueError(f"题目 {item_id} 的作答版本与冻结题库不一致")
        scoring_key = item["scoring_key"]
        if option_id not in scoring_key:
            raise ValueError(f"题目 {item_id} 选择了 scoring_key 中不存在的选项")
        rows.append(
            {
                "respondent_id": respondent_id,
                "tier_id": str(tier_id),
                "item_id": item_id,
                "item_version": expected_version,
                "dimension_id": item.get("target_dimension_id"),
                "context_category": item.get("context_category"),
                "raw_display_option_id": record.get("raw_display_option_id"),
                "selected_option_id": option_id,
                "score": float(scoring_key[option_id]),
            }
        )
    long_frame = pd.DataFrame(rows)
    if long_frame.empty:
        raise ValueError("分档SJT作答为空")
    respondent_ids = sorted(long_frame["respondent_id"].unique())
    if len(respondent_ids) != expected_respondent_count:
        raise ValueError("SJT 实际被试数与 manifest 不一致")
    expected_records = expected_respondent_count * len(item_order)
    if len(long_frame) != expected_records:
        raise ValueError(
            f"分档SJT作答不完整：应有 {expected_records} 条，实际 {len(long_frame)} 条"
        )
    wide = long_frame.pivot(
        index="respondent_id", columns="item_id", values="score"
    ).reindex(index=respondent_ids, columns=item_order)
    if wide.isna().any().any():
        raise ValueError("SJT 被试×题目矩阵存在缺失作答")
    tier_wide: dict[str, pd.DataFrame] = {}
    for tier_id in tier_ids:
        ids = sorted(
            long_frame.loc[long_frame["tier_id"] == tier_id, "respondent_id"].unique()
        )
        if not ids:
            raise ValueError(f"分数档 {tier_id} 没有作答")
        tier_wide[tier_id] = wide.loc[ids]
    return long_frame, wide, tier_wide


def _score_profile_frame(
    payload: Mapping[str, Any],
    *,
    expected_respondents: Sequence[str],
    score_specs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("score_profiles.json 缺少 profiles")
    dimension_ids = [str(spec["dimension_id"]) for spec in score_specs]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, Mapping):
            raise ValueError("score_profiles.json 包含无效 profile")
        respondent_id = profile.get("respondent_id")
        values = profile.get("score_values")
        if (
            not isinstance(respondent_id, str)
            or not respondent_id
            or respondent_id in seen
            or not isinstance(values, Mapping)
            or set(values) != set(dimension_ids)
        ):
            raise ValueError("score_profiles.json 的被试或维度集合无效")
        row = {"respondent_id": respondent_id}
        for dimension_id in dimension_ids:
            value = values[dimension_id]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{respondent_id} 的 {dimension_id} 分数无效")
            row[dimension_id] = float(value)
        rows.append(row)
        seen.add(respondent_id)
    frame = pd.DataFrame(rows).set_index("respondent_id")
    if sorted(frame.index) != sorted(expected_respondents):
        raise ValueError("分数档案与SJT被试集合不一致")
    return frame.reindex(index=list(expected_respondents), columns=dimension_ids)


def _frame_item_metrics(
    *,
    item_order: Sequence[str],
    items: Mapping[str, Mapping[str, Any]],
    response_wide: pd.DataFrame,
    score_frame: pd.DataFrame,
    score_specs: Sequence[Mapping[str, Any]],
    tier_ids: Sequence[Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Calculate CITC, target rho, and split VTS for one respondent frame."""

    specs_by_id: dict[str, dict[str, Any]] = {
        str(spec["dimension_id"]): dict(spec) for spec in score_specs
    }
    results: dict[str, dict[str, Any]] = {}
    facet_columns: dict[str, list[str]] = {}
    for item_id in item_order:
        facet_columns.setdefault(str(items[item_id]["target_dimension_id"]), []).append(item_id)
    for item_id in item_order:
        item = items[item_id]
        target_id = str(item["target_dimension_id"])
        target_spec = specs_by_id.get(target_id)
        if target_spec is None:
            raise ValueError(f"题目 {item_id} 缺少目标分数 {target_id}")
        same_facet_items = facet_columns[target_id]
        rest_score = response_wide[same_facet_items].sum(axis=1) - response_wide[item_id]
        citc = _pearson(response_wide[item_id], rest_score)
        association = _conditional_spearman if tier_ids is not None else _spearman
        target = (
            association(score_frame[target_id], response_wide[item_id], tier_ids)
            if tier_ids is not None
            else association(score_frame[target_id], response_wide[item_id])
        )
        same_domain_rows: list[dict[str, Any]] = []
        cross_domain_rows: list[dict[str, Any]] = []
        for spec in score_specs:
            dimension_id = str(spec["dimension_id"])
            if spec.get("level") != "facet" or dimension_id == target_id:
                continue
            non_target_association = (
                association(score_frame[dimension_id], response_wide[item_id], tier_ids)
                if tier_ids is not None
                else association(score_frame[dimension_id], response_wide[item_id])
            )
            row = {
                "dimension_id": dimension_id,
                "facet_name": spec.get("facet_name"),
                "facet_name_en": spec.get("facet_name_en"),
                "domain_id": spec.get("domain_id"),
                "domain_name": spec.get("domain_name"),
                "definition": spec.get("definition"),
                "high_behavior": spec.get("high_behavior"),
                "low_behavior": spec.get("low_behavior"),
                **non_target_association,
            }
            if spec.get("domain_id") == target_spec.get("domain_id"):
                same_domain_rows.append(row)
            else:
                cross_domain_rows.append(row)

        def vts_group(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
            estimable = [row for row in rows if row.get("rho") is not None]
            largest = (
                max(estimable, key=lambda row: float(row["rho"]))
                if estimable else None
            )
            largest_rho = float(largest["rho"]) if largest else None
            target_rho = target.get("rho")
            margin = (
                float(target_rho) - largest_rho
                if target_rho is not None and largest_rho is not None else None
            )
            return {
                "non_target_spearman": rows,
                "max_non_target_rho": largest_rho,
                "largest_non_target_dimension_id": (
                    largest.get("dimension_id") if largest else None
                ),
                "largest_non_target_facet_name": (
                    largest.get("facet_name") if largest else None
                ),
                "largest_non_target_domain_id": (
                    largest.get("domain_id") if largest else None
                ),
                "largest_non_target_rho": largest.get("rho") if largest else None,
                "largest_non_target_conditional_rho": (
                    largest.get("rho") if largest and tier_ids is not None else None
                ),
                "specificity_margin": margin,
                "margin_threshold": threshold,
                "passes": margin is not None and margin >= threshold,
                "largest_non_target_facet": deepcopy(largest) if largest else None,
            }

        target_rho = target.get("rho")
        same_domain = vts_group(same_domain_rows, SAME_DOMAIN_VTS_THRESHOLD)
        cross_domain = vts_group(cross_domain_rows, CROSS_DOMAIN_VTS_THRESHOLD)
        citc_pass = citc.get("r") is not None and citc["r"] >= CITC_REVISION_THRESHOLD
        target_rho_pass = (
            target_rho is not None and target_rho >= TARGET_RHO_THRESHOLD
        )
        results[item_id] = {
            "facet_citc": {
                **citc,
                "threshold": CITC_REVISION_THRESHOLD,
                "passes": citc_pass,
            },
            "virtual_target_specificity": {
                "target_dimension_id": target_id,
                "target_spearman": target,
                "conditional_target_spearman": target if tier_ids is not None else None,
                "correlation_scope": (
                    "aggregate_conditional_on_tier_id"
                    if tier_ids is not None else "within_tier_diagnostic"
                ),
                "target_rho_threshold": TARGET_RHO_THRESHOLD,
                "target_rho_pass": target_rho_pass,
                "same_domain_non_target": same_domain,
                "cross_domain_non_target": cross_domain,
                "passes": (
                    target_rho_pass and same_domain["passes"] and cross_domain["passes"]
                ),
            },
            "citc_pass": citc_pass,
            "target_rho_pass": target_rho_pass,
            "same_domain_vts_pass": bool(same_domain["passes"]),
            "cross_domain_vts_pass": bool(cross_domain["passes"]),
            "passes": (
                citc_pass and target_rho_pass
                and bool(same_domain["passes"])
                and bool(cross_domain["passes"])
            ),
        }
    return results


def _virtual_item_metrics(
    *,
    item_order: Sequence[str],
    items: Mapping[str, Mapping[str, Any]],
    response_wide: pd.DataFrame,
    tier_wide: Mapping[str, pd.DataFrame],
    score_frame: pd.DataFrame,
    score_specs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    tier_ids = pd.Series(index=response_wide.index, dtype="object")
    for tier_id, frame in tier_wide.items():
        tier_ids.loc[frame.index] = tier_id
    if tier_ids.isna().any():
        raise ValueError("合并作答中存在无法映射到tier_id的被试")
    aggregate = _frame_item_metrics(
        item_order=item_order,
        items=items,
        response_wide=response_wide,
        score_frame=score_frame,
        score_specs=score_specs,
        tier_ids=tier_ids.to_numpy(),
    )
    for item_id in item_order:
        aggregate[item_id]["per_tier_metrics"] = {}
    for tier_id, frame in tier_wide.items():
        tier_metrics = _frame_item_metrics(
            item_order=item_order,
            items=items,
            response_wide=frame,
            score_frame=score_frame.loc[frame.index],
            score_specs=score_specs,
        )
        for item_id in item_order:
            aggregate[item_id]["per_tier_metrics"][tier_id] = tier_metrics[item_id]
    return aggregate


def _score_neo(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_respondents: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        respondent_id = record.get("respondent_id")
        item_id = record.get("item_id")
        dimension = record.get("dimension_code")
        raw = record.get("raw_response")
        direction = record.get("scoring_direction")
        if not isinstance(respondent_id, str) or not respondent_id:
            raise ValueError("Neo-FFI 作答包含无效 respondent_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("Neo-FFI 作答包含无效 item_id")
        key = (respondent_id, item_id)
        if key in seen:
            raise ValueError(f"Neo-FFI 作答包含重复记录：{key}")
        seen.add(key)
        if dimension not in {"N", "E", "O", "A", "C"}:
            raise ValueError(f"Neo-FFI 包含未知维度：{dimension!r}")
        if (
            not isinstance(raw, int)
            or isinstance(raw, bool)
            or raw < 1
            or raw > 5
        ):
            raise ValueError(f"Neo-FFI 题目 {item_id} 的原始作答无效")
        if direction not in {"+", "-"}:
            raise ValueError(f"Neo-FFI 题目 {item_id} 的计分方向无效")
        rows.append(
            {
                "respondent_id": respondent_id,
                "dimension_code": dimension,
                "item_id": item_id,
                "raw_response": raw,
                "scoring_direction": direction,
                "score": raw if direction == "+" else 6 - raw,
            }
        )
    long_frame = pd.DataFrame(rows)
    respondent_ids = sorted(long_frame["respondent_id"].unique())
    if respondent_ids != sorted(expected_respondents):
        raise ValueError("SJT 与 Neo-FFI 的被试集合不一致")
    counts = long_frame.groupby(
        ["respondent_id", "dimension_code"]
    ).size()
    if len(counts) != len(respondent_ids) * 5 or not (counts == 12).all():
        raise ValueError("每名被试的每个 Neo-FFI 维度必须正好有12题")
    dimension_scores = (
        long_frame.groupby(["respondent_id", "dimension_code"])["score"]
        .mean()
        .unstack("dimension_code")
        .reindex(index=respondent_ids, columns=["N", "E", "O", "A", "C"])
    )
    return long_frame, dimension_scores


def _item_statistics(
    sjt_long: pd.DataFrame,
    sjt_wide: pd.DataFrame,
    *,
    item_order: Sequence[str],
    items: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    facet_columns: dict[str, list[str]] = {}
    for item_id in item_order:
        dimension_id = str(items[item_id]["target_dimension_id"])
        facet_columns.setdefault(dimension_id, []).append(item_id)
    statistics_by_item: dict[str, dict[str, Any]] = {}
    flat_rows: list[dict[str, Any]] = []
    option_rows: list[dict[str, Any]] = []

    for item_id in item_order:
        item = items[item_id]
        dimension_id = str(item["target_dimension_id"])
        scores = sjt_wide[item_id]
        scoring_key = item["scoring_key"]
        theoretical_minimum = float(min(scoring_key.values()))
        theoretical_maximum = float(max(scoring_key.values()))
        difficulty = standardized_item_difficulty(
            scores,
            theoretical_minimum,
            theoretical_maximum,
        )
        same_facet_items = facet_columns[dimension_id]
        facet_rest_score = sjt_wide[same_facet_items].sum(axis=1) - scores
        corrected = _pearson(scores, facet_rest_score)

        item_responses = sjt_long[sjt_long["item_id"] == item_id]
        option_counts = item_responses[
            "selected_option_id"
        ].value_counts()
        option_statistics: dict[str, dict[str, Any]] = {}
        for option_id in scoring_key:
            count = int(option_counts.get(option_id, 0))
            rate = count / len(item_responses)
            option_result = {
                "count": count,
                "rate": float(rate),
                "score": float(scoring_key[option_id]),
                "passes_minimum_rate": rate >= MINIMUM_OPTION_RATE,
            }
            option_statistics[str(option_id)] = option_result
            option_rows.append(
                {
                    "item_id": item_id,
                    "option_id": option_id,
                    **option_result,
                }
            )

        minimum_option_rate = min(
            option["rate"] for option in option_statistics.values()
        )
        effective_option_count = sum(
            option["rate"] >= MINIMUM_OPTION_RATE
            for option in option_statistics.values()
        )
        difficulty_pass = (
            DIFFICULTY_LOWER_BOUND
            <= difficulty
            <= DIFFICULTY_UPPER_BOUND
        )
        citc_pass = (
            corrected["r"] is not None
            and corrected["r"] >= CITC_REVISION_THRESHOLD
        )
        option_rate_pass = effective_option_count >= 3
        qualified = difficulty_pass and citc_pass and option_rate_pass

        result = {
            "item_id": item_id,
            "dimension_id": dimension_id,
            "context_category": item.get("context_category"),
            "n": int(scores.count()),
            "mean": float(scores.mean()),
            "standard_deviation": float(scores.std(ddof=1)),
            "observed_minimum": float(scores.min()),
            "observed_maximum": float(scores.max()),
            "theoretical_minimum": theoretical_minimum,
            "theoretical_maximum": theoretical_maximum,
            "difficulty": difficulty,
            "facet_corrected_item_total_correlation": corrected,
            "option_statistics": option_statistics,
            "minimum_option_rate": minimum_option_rate,
            "effective_option_count": effective_option_count,
            "qualification": {
                "difficulty_pass": difficulty_pass,
                "citc_pass": citc_pass,
                "option_rate_pass": option_rate_pass,
                "qualified": qualified,
            },
        }
        statistics_by_item[item_id] = _json_safe(result)
        flat_rows.append(
            {
                key: value
                for key, value in {
                    "item_id": item_id,
                    "dimension_id": dimension_id,
                    "context_category": item.get("context_category"),
                    "n": result["n"],
                    "mean": result["mean"],
                    "standard_deviation": result["standard_deviation"],
                    "observed_minimum": result["observed_minimum"],
                    "observed_maximum": result["observed_maximum"],
                    "theoretical_minimum": theoretical_minimum,
                    "theoretical_maximum": theoretical_maximum,
                    "difficulty": difficulty,
                    "citc_r": corrected["r"],
                    "citc_p": corrected["p_value"],
                    "minimum_option_rate": minimum_option_rate,
                    "effective_option_count": effective_option_count,
                    "difficulty_pass": difficulty_pass,
                    "citc_pass": citc_pass,
                    "option_rate_pass": option_rate_pass,
                    "qualified": qualified,
                }.items()
            }
        )
    return (
        statistics_by_item,
        pd.DataFrame(flat_rows),
        pd.DataFrame(option_rows),
    )


def _score_matched_sjt(
    records: Sequence[Mapping[str, Any]],
    *,
    item_order: Sequence[str],
    items: Mapping[str, Mapping[str, Any]],
    expected_respondent_count: int,
    condition_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Score one response per item and validate exact matching across facet groups."""

    expected_ids = tuple(str(value) for value in (condition_ids or MATCHED_CONDITION_IDS))
    if "target" not in expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("matched condition_ids 必须包含唯一的 target group")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    condition_subjects: dict[str, set[str]] = {condition_id: set() for condition_id in expected_ids}
    for record in records:
        respondent_id = str(record.get("respondent_id") or "")
        condition_id = str(record.get("condition_id") or "")
        matched_subject_id = str(record.get("matched_subject_id") or "")
        item_id = str(record.get("item_id") or "")
        option_id = record.get("selected_option_id")
        if condition_id not in expected_ids or not respondent_id or not matched_subject_id:
            raise ValueError("SJT 作答缺少有效 condition_id 或 matched_subject_id")
        if item_id not in items:
            raise ValueError(f"SJT 作答引用未知题目：{item_id!r}")
        key = (condition_id, matched_subject_id, respondent_id, item_id)
        if key in seen:
            raise ValueError(f"SJT 作答包含重复记录：{key}")
        seen.add(key)
        item = items[item_id]
        if record.get("item_version") != item.get("version"):
            raise ValueError(f"题目 {item_id} 的作答版本与冻结题库不一致")
        scoring_key = item["scoring_key"]
        if option_id not in scoring_key:
            raise ValueError(f"题目 {item_id} 选择了 scoring_key 中不存在的选项")
        condition_subjects[condition_id].add(matched_subject_id)
        rows.append({
            "respondent_id": respondent_id,
            "condition_id": condition_id,
            "arm_id": record.get("arm_id") or ("target" if condition_id == "target" else condition_id.split("__", 1)[0]),
            "group_id": record.get("group_id") or ("target" if condition_id == "target" else condition_id.split("__", 1)[-1]),
            "matched_subject_id": matched_subject_id,
            "item_id": item_id,
            "item_version": item.get("version"),
            "dimension_id": item.get("target_dimension_id"),
            "context_category": item.get("context_category"),
            "raw_display_option_id": record.get("raw_display_option_id"),
            "selected_option_id": str(option_id),
            "score": float(scoring_key[option_id]),
            "active_score": float(record.get("active_score")) if record.get("active_score") is not None else np.nan,
        })
    long_frame = pd.DataFrame(rows)
    if long_frame.empty:
        raise ValueError("匹配 facet SJT 作答为空")
    for condition_id, subjects in condition_subjects.items():
        if len(subjects) != expected_respondent_count:
            raise ValueError(f"条件 {condition_id} 的被试数与 manifest 不一致")
    target_subjects = set(condition_subjects["target"])
    if any(set(subjects) != target_subjects for subjects in condition_subjects.values()):
        raise ValueError("所有 facet group 的 matched_subject_id 集合必须完全一致")
    expected_records = expected_respondent_count * len(item_order) * len(expected_ids)
    if len(long_frame) != expected_records:
        raise ValueError(f"匹配 facet SJT 作答不完整：应有 {expected_records} 条，实际 {len(long_frame)} 条")
    condition_wide: dict[str, pd.DataFrame] = {}
    condition_scores: dict[str, pd.Series] = {}
    for condition_id in expected_ids:
        frame = long_frame[long_frame["condition_id"] == condition_id]
        wide = frame.pivot(index="matched_subject_id", columns="item_id", values="score").reindex(columns=list(item_order))
        if wide.isna().any().any() or len(wide) != expected_respondent_count:
            raise ValueError(f"条件 {condition_id} 的被试×题目矩阵存在缺失作答")
        condition_wide[condition_id] = wide.sort_index()
        score_series = frame.drop_duplicates("matched_subject_id").set_index("matched_subject_id")["active_score"].reindex(wide.index)
        if score_series.isna().any():
            raise ValueError(f"条件 {condition_id} 缺少匹配 facet 分数")
        condition_scores[condition_id] = score_series.astype(float)
    return long_frame, condition_wide, condition_scores


def _matched_item_metrics(
    *,
    item_order: Sequence[str],
    items: Mapping[str, Mapping[str, Any]],
    condition_wide: Mapping[str, pd.DataFrame],
    condition_scores: Mapping[str, pd.Series],
    conditions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Calculate target-arm CITC and signed MAX VTS across fixed arms."""

    group_rows = flatten_matched_condition_groups(conditions)
    condition_rows = {str(row.get("condition_id")): dict(row) for row in group_rows}
    target_row = condition_rows.get("target") or {}
    arm_group_ids: dict[str, list[str]] = {arm_id: [] for arm_id in MATCHED_CONDITION_IDS}
    for row in group_rows:
        arm_group_ids.setdefault(str(row.get("arm_id")), []).append(str(row.get("condition_id")))
    target_wide = condition_wide["target"]
    facet_columns: dict[str, list[str]] = {}
    for item_id in item_order:
        facet_columns.setdefault(str(items[item_id]["target_dimension_id"]), []).append(item_id)
    metrics: dict[str, dict[str, Any]] = {}
    for item_id in item_order:
        item = items[item_id]
        target_id = str(item["target_dimension_id"])
        same_items = facet_columns[target_id]
        citc = _pearson(target_wide[item_id], target_wide[same_items].sum(axis=1) - target_wide[item_id])
        rhos = {
            condition_id: _spearman(
                condition_scores[condition_id],
                condition_wide[condition_id][item_id],
            )
            for condition_id in condition_rows
        }
        target_rho = rhos["target"].get("rho")

        def vts_group(arm_id: str, threshold: float) -> dict[str, Any]:
            rows = []
            for condition_id in arm_group_ids.get(arm_id, []):
                metadata = condition_rows.get(condition_id, {})
                rows.append({
                    **metadata,
                    "condition_id": condition_id,
                    "arm_id": arm_id,
                    "rho": rhos[condition_id].get("rho"),
                })
            estimable = [row for row in rows if row.get("rho") is not None]
            largest = max(estimable, key=lambda row: float(row["rho"])) if estimable else None
            max_rho = float(largest["rho"]) if largest else None
            margin = float(target_rho) - max_rho if target_rho is not None and max_rho is not None else None
            return {
                "non_target_spearman": rows,
                "max_non_target_rho": max_rho,
                "largest_non_target_dimension_id": largest.get("dimension_id") if largest else None,
                "largest_non_target_facet_name": largest.get("facet_name") if largest else None,
                "largest_non_target_domain_id": largest.get("domain_id") if largest else None,
                "largest_non_target_rho": max_rho,
                "selected_non_target_condition_id": largest.get("condition_id") if largest else None,
                "selected_non_target_group_id": largest.get("group_id") if largest else None,
                "selected_non_target_facet": deepcopy(largest) if largest else None,
                "specificity_margin": margin,
                "margin_threshold": threshold,
                "passes": margin is not None and margin >= threshold,
                "largest_non_target_facet": deepcopy(largest) if largest else None,
            }

        same_domain = vts_group("same_domain", SAME_DOMAIN_VTS_THRESHOLD)
        cross_domain = vts_group("cross_domain", CROSS_DOMAIN_VTS_THRESHOLD)
        same_rho = same_domain.get("max_non_target_rho")
        cross_rho = cross_domain.get("max_non_target_rho")
        same_vts = same_domain.get("specificity_margin")
        cross_vts = cross_domain.get("specificity_margin")
        metrics[item_id] = {
            "facet_citc": {**citc, "threshold": CITC_REVISION_THRESHOLD, "passes": citc.get("r") is not None and citc["r"] >= CITC_REVISION_THRESHOLD, "condition_id": "target", "filtering_authority": True},
            "virtual_target_specificity": {
                "target_dimension_id": target_id,
                "target_spearman": rhos["target"],
                "rho_target": rhos["target"],
                "rho_same_domain": {"rho": same_rho, "selected_group_id": same_domain.get("selected_non_target_group_id"), "selected_condition_id": same_domain.get("selected_non_target_condition_id"), "groups": same_domain.get("non_target_spearman", [])},
                "rho_cross_domain": {"rho": cross_rho, "selected_group_id": cross_domain.get("selected_non_target_group_id"), "selected_condition_id": cross_domain.get("selected_non_target_condition_id"), "groups": cross_domain.get("non_target_spearman", [])},
                "correlation_scope": "matched_facet_condition",
                "conditioning_variable": "condition_id",
                "target_rho_threshold": TARGET_RHO_THRESHOLD,
                "target_rho_pass": target_rho is not None and target_rho >= TARGET_RHO_THRESHOLD,
                "same_domain_non_target": {
                    **same_domain,
                    "arm_id": "same_domain",
                    "filtering_authority": True,
                },
                "cross_domain_non_target": {
                    **cross_domain,
                    "arm_id": "cross_domain",
                    "filtering_authority": True,
                },
            },
            "per_condition_metrics": {
                condition_id: {
                    "condition_id": condition_id,
                    "arm_id": condition_rows[condition_id].get("arm_id"),
                    "group_id": condition_rows[condition_id].get("group_id"),
                    "rho": rhos[condition_id],
                    "citc": _pearson(
                        condition_wide[condition_id][item_id],
                        condition_wide[condition_id][same_items].sum(axis=1) - condition_wide[condition_id][item_id],
                    ),
                    "filtering_authority": condition_id == "target",
                }
                for condition_id in condition_rows
            },
            "citc_pass": citc.get("r") is not None and citc["r"] >= CITC_REVISION_THRESHOLD,
            "target_rho_pass": target_rho is not None and target_rho >= TARGET_RHO_THRESHOLD,
            "same_domain_vts_pass": same_vts is not None and same_vts >= SAME_DOMAIN_VTS_THRESHOLD,
            "cross_domain_vts_pass": cross_vts is not None and cross_vts >= CROSS_DOMAIN_VTS_THRESHOLD,
        }
        metric = metrics[item_id]
        metric["passes"] = bool(metric["citc_pass"] and metric["target_rho_pass"] and metric["same_domain_vts_pass"] and metric["cross_domain_vts_pass"])
        metric["qualified"] = metric["passes"]
    return metrics


def _target_option_gradient(
    *,
    target_long: pd.DataFrame,
    item: Mapping[str, Any],
    target_scores: pd.Series,
) -> dict[str, Any]:
    """Summarize target-arm facet means by scored option and adjacent failures."""

    item_id = str(item["item_id"])
    frame = target_long[target_long["item_id"] == item_id].copy()
    scoring_key = {str(key): float(value) for key, value in (item.get("scoring_key") or {}).items()}
    result_options: list[dict[str, Any]] = []
    means: dict[str, float | None] = {}
    ordered_scoring = sorted(scoring_key.items(), key=lambda pair: (pair[1], pair[0]))
    for option_id, option_score in ordered_scoring:
        ids = frame.loc[frame["selected_option_id"].astype(str) == option_id, "matched_subject_id"]
        values = target_scores.reindex(ids).dropna()
        mean = float(values.mean()) if len(values) else None
        means[option_id] = mean
        result_options.append({"option_id": option_id, "score": option_score, "n": int(len(values)), "target_facet_mean": mean, "standard_error": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) >= 2 else None})
    failures: list[dict[str, Any]] = []
    option_ids = [option_id for option_id, _ in ordered_scoring]
    estimable = True
    for option_id in option_ids:
        if means[option_id] is None:
            estimable = False
    for lower, higher in zip(option_ids, option_ids[1:]):
        low_mean, high_mean = means[lower], means[higher]
        passed = low_mean is not None and high_mean is not None and low_mean < high_mean
        if not passed:
            failures.append({"lower_option_id": lower, "higher_option_id": higher, "lower_mean": low_mean, "higher_mean": high_mean, "direction": "加强高等级选项目标线索，或削弱低等级选项非目标/过强目标线索"})
    return {"estimable": bool(estimable), "options": result_options, "adjacent_comparisons": [{"lower_option_id": lower, "higher_option_id": higher, "lower_mean": means[lower], "higher_mean": means[higher], "passes": means[lower] is not None and means[higher] is not None and means[lower] < means[higher]} for lower, higher in zip(option_ids, option_ids[1:])], "failed_adjacent_pairs": failures, "passes": bool(estimable and not failures), "filtering_authority": False}


def _matched_option_diagnostics(
    *,
    sjt_long: pd.DataFrame,
    items: Mapping[str, Mapping[str, Any]],
    item_order: Sequence[str],
    condition_scores: Mapping[str, pd.Series],
    gradients: Mapping[str, Mapping[str, Any]],
    conditions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    condition_groups = flatten_matched_condition_groups(conditions)
    condition_metadata = {
        str(row.get("condition_id")): dict(row)
        for row in condition_groups
        if isinstance(row, Mapping)
    }
    condition_ids = tuple(condition_metadata)

    def mean_and_se(values: pd.Series) -> tuple[float | None, float | None]:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return None, None
        mean = float(numeric.mean())
        standard_error = (
            float(numeric.std(ddof=1) / np.sqrt(len(numeric)))
            if len(numeric) >= 2
            else None
        )
        return mean, standard_error

    def arm_difference(
        item_id: str,
        item: Mapping[str, Any],
        comparator_id: str,
    ) -> dict[str, Any]:
        target = sjt_long[
            (sjt_long["item_id"] == item_id)
            & (sjt_long["condition_id"] == "target")
        ][
            [
                "matched_subject_id",
                "selected_option_id",
                "score",
                "active_score",
            ]
        ].rename(
            columns={
                "selected_option_id": "target_option_id",
                "score": "target_item_score",
                "active_score": "target_facet_score",
            }
        )
        comparator = sjt_long[
            (sjt_long["item_id"] == item_id)
            & (sjt_long["condition_id"] == comparator_id)
        ][
            [
                "matched_subject_id",
                "selected_option_id",
                "score",
                "active_score",
            ]
        ].rename(
            columns={
                "selected_option_id": "comparator_option_id",
                "score": "comparator_item_score",
                "active_score": "comparator_facet_score",
            }
        )
        paired = target.merge(
            comparator,
            on="matched_subject_id",
            how="inner",
            validate="one_to_one",
        ).sort_values("matched_subject_id")
        score_sequences_matched = bool(
            len(paired) == len(target) == len(comparator)
            and np.allclose(
                paired["target_facet_score"].to_numpy(dtype=float),
                paired["comparator_facet_score"].to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
            )
        )
        estimable = bool(
            score_sequences_matched
            and len(paired) >= MINIMUM_OPTION_DIAGNOSTIC_GROUP_N
            and paired["target_facet_score"].nunique(dropna=True) > 1
        )
        ordered_options = sorted(
            (
                (str(option_id), float(score))
                for option_id, score in (item.get("scoring_key") or {}).items()
            ),
            key=lambda pair: (pair[1], pair[0]),
        )
        high_option_ids = {
            option_id for option_id, _ in ordered_options[-2:]
        }

        def paired_summary(frame: pd.DataFrame, label: str) -> dict[str, Any]:
            group_n = len(frame)
            target_mean = (
                float(frame["target_item_score"].mean()) if group_n else None
            )
            comparator_mean = (
                float(frame["comparator_item_score"].mean()) if group_n else None
            )
            option_rows: list[dict[str, Any]] = []
            for option_id, option_score in ordered_options:
                target_count = int(
                    (frame["target_option_id"].astype(str) == option_id).sum()
                )
                comparator_count = int(
                    (frame["comparator_option_id"].astype(str) == option_id).sum()
                )
                target_rate = float(target_count / group_n) if group_n else None
                comparator_rate = (
                    float(comparator_count / group_n) if group_n else None
                )
                option_rows.append(
                    {
                        "option_id": option_id,
                        "score": option_score,
                        "target_count": target_count,
                        "target_rate": target_rate,
                        "comparator_count": comparator_count,
                        "comparator_rate": comparator_rate,
                        "target_minus_comparator_rate": (
                            target_rate - comparator_rate
                            if target_rate is not None and comparator_rate is not None
                            else None
                        ),
                    }
                )
            target_high_rate = (
                float(frame["target_option_id"].isin(high_option_ids).mean())
                if group_n
                else None
            )
            comparator_high_rate = (
                float(frame["comparator_option_id"].isin(high_option_ids).mean())
                if group_n
                else None
            )
            return {
                "score_band": label,
                "group_n": group_n,
                "target_mean_item_score": target_mean,
                "comparator_mean_item_score": comparator_mean,
                "target_minus_comparator_mean_item_score": (
                    target_mean - comparator_mean
                    if target_mean is not None and comparator_mean is not None
                    else None
                ),
                "target_high_option_rate": target_high_rate,
                "comparator_high_option_rate": comparator_high_rate,
                "target_minus_comparator_high_option_rate": (
                    target_high_rate - comparator_high_rate
                    if target_high_rate is not None
                    and comparator_high_rate is not None
                    else None
                ),
                "options": option_rows,
            }

        score_bands: list[dict[str, Any]] = []
        if estimable:
            try:
                paired = paired.copy()
                paired["score_band"] = pd.qcut(
                    paired["target_facet_score"],
                    q=3,
                    labels=["low", "middle", "high"],
                    duplicates="drop",
                )
                labels = {
                    str(value)
                    for value in paired["score_band"].dropna().astype(str).unique()
                }
                if labels == {"low", "middle", "high"}:
                    score_bands = [
                        paired_summary(
                            paired[paired["score_band"].astype(str) == label],
                            label,
                        )
                        for label in ("low", "middle", "high")
                    ]
                else:
                    estimable = False
            except (ValueError, TypeError):
                estimable = False

        overall = paired_summary(paired, "all")
        overall.update(
            {
                "same_option_count": int(
                    (
                        paired["target_option_id"].astype(str)
                        == paired["comparator_option_id"].astype(str)
                    ).sum()
                ),
                "same_option_rate": (
                    float(
                        (
                            paired["target_option_id"].astype(str)
                            == paired["comparator_option_id"].astype(str)
                        ).mean()
                    )
                    if len(paired)
                    else None
                ),
            }
        )
        metadata = condition_metadata.get(comparator_id) or {}
        return {
            "comparison_id": f"target_vs_{comparator_id}",
            "target_condition_id": "target",
            "comparator_condition_id": comparator_id,
            "comparator_role": metadata.get("role"),
            "comparator_arm_id": metadata.get("arm_id"),
            "comparator_group_id": metadata.get("group_id"),
            "comparator_dimension_id": metadata.get("dimension_id"),
            "comparator_facet_name": metadata.get("facet_name"),
            "matched_subject_count": len(paired),
            "score_sequences_matched": score_sequences_matched,
            "estimable": estimable,
            "reason": (
                None
                if estimable
                else "匹配分数序列、样本量或分数方差不足；差异仅展示，不得用于返修推断"
            ),
            "overall": overall,
            "by_score_band": score_bands,
            "filtering_authority": False,
            "diagnostic_use": "localization_only",
        }

    diagnostics: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for item_id in item_order:
        item = items[item_id]
        item_diag: dict[str, Any] = {
            "version": OPTION_CHOICE_DIAGNOSTICS_VERSION,
            "filtering_authority": False,
            "all": None,
            "by_condition": [],
            "target_option_gradient": gradients[item_id],
            "arm_difference_diagnostics": {
                "filtering_authority": False,
                "diagnostic_use": "localization_only",
                "comparisons": [
                    arm_difference(item_id, item, condition_id)
                    for condition_id in condition_ids
                    if condition_id != "target"
                ],
            },
        }
        all_group = sjt_long[sjt_long["item_id"] == item_id]
        condition_option_rows: dict[str, dict[str, dict[str, Any]]] = {}
        for scope, condition_id, group in [("all", None, all_group), *[("condition", cid, sjt_long[(sjt_long["item_id"] == item_id) & (sjt_long["condition_id"] == cid)]) for cid in condition_ids]]:
            counts = group["selected_option_id"].astype(str).value_counts()
            group_n = len(group)
            options = []
            metadata = condition_metadata.get(str(condition_id)) or {}
            for option_id, score in sorted(
                (item.get("scoring_key") or {}).items(),
                key=lambda pair: (float(pair[1]), str(pair[0])),
            ):
                option_id = str(option_id)
                count = int(counts.get(option_id, 0))
                selected_scores = group.loc[
                    group["selected_option_id"].astype(str) == option_id,
                    "active_score",
                ]
                facet_mean, facet_standard_error = mean_and_se(selected_scores)
                row = {
                    "option_id": option_id,
                    "score": float(score),
                    "selection_count": count,
                    "selection_rate": float(count / group_n) if group_n else None,
                    "facet_mean": facet_mean,
                    "facet_standard_error": facet_standard_error,
                }
                options.append(row)
                rows.append({"item_id": item_id, "grouping_scope": scope, "condition_id": condition_id, "arm_id": metadata.get("arm_id") if condition_id is not None else None, "group_id": metadata.get("group_id") if condition_id is not None else None, "matched_group": "all", "construct_role": metadata.get("role") if condition_id is not None else "pooled_matched_active_facet", "facet_id": metadata.get("dimension_id") if condition_id is not None else None, "group_n": group_n, "option_id": option_id, "option_score": float(score), "selection_count": count, "selection_rate": row["selection_rate"], "facet_mean": facet_mean, "facet_standard_error": facet_standard_error, "estimable": group_n >= MINIMUM_OPTION_DIAGNOSTIC_GROUP_N, "filtering_authority": False})
                if condition_id is not None:
                    condition_option_rows.setdefault(str(condition_id), {})[option_id] = {
                        "n": count,
                        "mean_score": facet_mean,
                        "standard_error": facet_standard_error,
                    }
            group_result = {
                "condition_id": condition_id,
                "arm_id": metadata.get("arm_id") if condition_id is not None else None,
                "group_id": metadata.get("group_id") if condition_id is not None else None,
                "construct_role": metadata.get("role") if condition_id is not None else "pooled_matched_active_facet",
                "facet_id": metadata.get("dimension_id") if condition_id is not None else None,
                "facet_name": metadata.get("facet_name") if condition_id is not None else None,
                "group_n": group_n,
                "options": options,
                "estimable": group_n >= MINIMUM_OPTION_DIAGNOSTIC_GROUP_N,
                "reason": (
                    None
                    if group_n >= MINIMUM_OPTION_DIAGNOSTIC_GROUP_N
                    else "该条件组少于10人；频率仅展示，不得用于返修推断"
                ),
            }
            if scope == "all":
                item_diag["all"] = group_result
            else:
                item_diag["by_condition"].append(group_result)
        paired_rows: list[dict[str, Any]] = []
        for option_id, score in sorted(
            ((str(key), float(value)) for key, value in (item.get("scoring_key") or {}).items()),
            key=lambda pair: (pair[1], pair[0]),
        ):
            target = condition_option_rows.get("target", {}).get(option_id, {})
            same_groups = [
                condition_option_rows.get(condition_id, {}).get(option_id, {})
                for condition_id in condition_ids
                if condition_metadata.get(condition_id, {}).get("arm_id") == "same_domain"
            ]
            cross_groups = [
                condition_option_rows.get(condition_id, {}).get(option_id, {})
                for condition_id in condition_ids
                if condition_metadata.get(condition_id, {}).get("arm_id") == "cross_domain"
            ]
            same = next((row for row in same_groups if row), {})
            cross = next((row for row in cross_groups if row), {})
            paired_rows.append(
                {
                    "option_id": option_id,
                    "score": score,
                    "target_n": target.get("n"),
                    "target_mean_score": target.get("mean_score"),
                    "target_standard_error": target.get("standard_error"),
                    "same_domain_n": same.get("n"),
                    "same_domain_mean_score": same.get("mean_score"),
                    "same_domain_standard_error": same.get("standard_error"),
                    "cross_domain_n": cross.get("n"),
                    "cross_domain_mean_score": cross.get("mean_score"),
                    "cross_domain_standard_error": cross.get("standard_error"),
                    "same_domain_groups": same_groups,
                    "cross_domain_groups": cross_groups,
                    "filtering_authority": False,
                }
            )
        item_diag["option_score_comparisons"] = paired_rows
        diagnostics[item_id] = item_diag
    return diagnostics, pd.DataFrame(rows)


def _apply_matched_quality(
    item_statistics: dict[str, dict[str, Any]],
    virtual_metrics: Mapping[str, Mapping[str, Any]],
    gradients: Mapping[str, Mapping[str, Any]],
    *,
    item_order: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    rows = []
    for item_id in item_order:
        item = item_statistics[item_id]
        metric = virtual_metrics[item_id]
        specificity = metric["virtual_target_specificity"]
        same = specificity["same_domain_non_target"]
        cross = specificity["cross_domain_non_target"]
        failed = []
        if not metric["citc_pass"]: failed.append("CITC")
        if not metric["target_rho_pass"]: failed.append("目标rho")
        if not metric["same_domain_vts_pass"]: failed.append("同域VTS")
        if not metric["cross_domain_vts_pass"]: failed.append("跨域VTS")
        qualified = bool(metric["passes"])
        item["target_option_gradient"] = _json_safe(gradients[item_id])
        item["virtual_screening_metrics"] = _json_safe(metric)
        item["qualification"] = {"citc_pass": metric["citc_pass"], "target_rho_pass": metric["target_rho_pass"], "same_domain_vts_pass": metric["same_domain_vts_pass"], "cross_domain_vts_pass": metric["cross_domain_vts_pass"], "qualified": qualified}
        item["quality_evaluation"] = {"version": MEASUREMENT_EVALUATION_VERSION, "quality_grade": "acceptable" if qualified else "needs_revision", "recommendation": "retain" if qualified else "revise", "facet_citc": metric["facet_citc"], "virtual_target_specificity": specificity, "per_condition_metrics": metric.get("per_condition_metrics") or {}, "diagnostic_flags": failed, "failed_gates": failed, "target_option_gradient_failed": not gradients[item_id].get("passes", False), "decision_rule": "资格仅使用目标臂CITC、三臂rho和两类VTS；每个非目标facet group单独估计并取臂内最大带符号rho；目标组选项梯度只触发返修。"}
        rows.append({"item_id": item_id, "metric_scope": "matched_conditions", "filtering_authority": True, "correlation_method": "ordinary_spearman", "conditioning_variable": "condition_id", "quality_grade": item["quality_evaluation"]["quality_grade"], "recommendation": item["quality_evaluation"]["recommendation"], "citc_r": metric["facet_citc"].get("r"), "rho_target": specificity["rho_target"].get("rho"), "rho_same_domain": specificity["rho_same_domain"].get("rho"), "rho_cross_domain": specificity["rho_cross_domain"].get("rho"), "same_domain_max_non_target_rho": same.get("max_non_target_rho"), "cross_domain_max_non_target_rho": cross.get("max_non_target_rho"), "same_domain_group_count": len(same.get("non_target_spearman") or []), "cross_domain_group_count": len(cross.get("non_target_spearman") or []), "same_domain_vts": same.get("specificity_margin"), "cross_domain_vts": cross.get("specificity_margin"), "failed_gates": "；".join(failed), "target_option_gradient_pass": gradients[item_id].get("passes"), "target_option_gradient_estimable": gradients[item_id].get("estimable")})
    return item_statistics, pd.DataFrame(rows)


def _run_matched_condition_analysis(
    *,
    state: PSJTState,
    manifest_path: Path,
    sjt_path: Path,
    response_manifest: Mapping[str, Any],
    item_order: list[str],
    items: Mapping[str, Mapping[str, Any]],
    expected_respondents: int,
) -> dict[str, Any]:
    """Persist the active fixed-three-arm, nested-facet-group screening analysis."""

    if response_manifest.get("schema_version") != MATCHED_CONDITION_SCHEMA_VERSION:
        raise ValueError(f"匹配 facet 分析需要 schema_version={MATCHED_CONDITION_SCHEMA_VERSION}")
    conditions = response_manifest.get("conditions")
    if not isinstance(conditions, list) or {str(row.get("condition_id")) for row in conditions if isinstance(row, Mapping)} != set(MATCHED_CONDITION_IDS):
        raise ValueError("作答 manifest 缺少固定三臂匹配条件")
    condition_groups = flatten_matched_condition_groups(conditions)
    condition_ids = tuple(str(row.get("condition_id")) for row in condition_groups)
    if "target" not in condition_ids or len(set(condition_ids)) != len(condition_ids):
        raise ValueError("作答 manifest 的 facet groups 缺少唯一 target 或存在重复 condition_id")
    expected_per_condition = response_manifest.get("sample_size_per_condition")
    if not isinstance(expected_per_condition, int) or expected_respondents != expected_per_condition * len(condition_ids):
        raise ValueError("facet group 作答人数与 sample_size_per_condition 不一致")
    sjt_records = _read_jsonl(sjt_path)
    sjt_long, condition_wide, condition_scores = _score_matched_sjt(
        sjt_records,
        item_order=item_order,
        items=items,
        expected_respondent_count=expected_per_condition,
        condition_ids=condition_ids,
    )
    virtual_metrics = _matched_item_metrics(
        item_order=item_order,
        items=items,
        condition_wide=condition_wide,
        condition_scores=condition_scores,
        conditions=conditions,
    )
    item_statistics, item_frame, _ = _item_statistics(
        sjt_long,
        pd.concat([condition_wide[c].assign(condition_id=c) for c in condition_ids]).drop(columns=["condition_id"]),
        item_order=item_order,
        items=items,
    )
    item_statistics, _ = _enrich_item_quality(item_statistics, item_order=item_order)
    target_long = sjt_long[sjt_long["condition_id"] == "target"]
    gradients = {
        item_id: _target_option_gradient(target_long=target_long, item=items[item_id], target_scores=condition_scores["target"])
        for item_id in item_order
    }
    item_statistics, item_quality_frame = _apply_matched_quality(
        item_statistics,
        virtual_metrics,
        gradients,
        item_order=item_order,
    )
    option_diagnostics, option_frame = _matched_option_diagnostics(
        sjt_long=sjt_long,
        items=items,
        item_order=item_order,
        condition_scores=condition_scores,
        gradients=gradients,
        conditions=conditions,
    )
    for item_id in item_order:
        item_statistics[item_id]["option_choice_diagnostics"] = option_diagnostics[item_id]
    target_wide = condition_wide["target"]
    scale_statistics = _scale_statistics(target_wide, items)
    analysis_round = int(state.get("psychometric_analysis_round") or 0) + 1
    respondent_rows = []
    for condition_id in condition_ids:
        frame = condition_wide[condition_id]
        scores = condition_scores[condition_id]
        for matched_subject_id in frame.index:
            group = next((entry for entry in condition_groups if entry.get("condition_id") == condition_id), {})
            row = {"respondent_id": f"{condition_id}-{matched_subject_id}", "condition_id": condition_id, "arm_id": group.get("arm_id"), "group_id": group.get("group_id"), "matched_subject_id": matched_subject_id, "active_score": float(scores.loc[matched_subject_id]), "sjt_total_score": float(frame.loc[matched_subject_id].mean())}
            respondent_rows.append(row)
    respondent_scores = pd.DataFrame(respondent_rows)
    target_rhos = [virtual_metrics[item_id]["virtual_target_specificity"]["rho_target"].get("rho") for item_id in item_order]
    citcs = [virtual_metrics[item_id]["facet_citc"].get("r") for item_id in item_order]
    same_vts = [virtual_metrics[item_id]["virtual_target_specificity"]["same_domain_non_target"].get("specificity_margin") for item_id in item_order]
    cross_vts = [virtual_metrics[item_id]["virtual_target_specificity"]["cross_domain_non_target"].get("specificity_margin") for item_id in item_order]
    validity_statistics = _json_safe({
        "evidence_scope": "exploratory_virtual_matched_facet_screening",
        "psychometric_analysis_round": analysis_round,
        "sampling_design": "matched_facet_conditions",
        "condition_ids": list(condition_ids),
        "arm_ids": list(MATCHED_CONDITION_IDS),
        "group_count": len(condition_ids),
        "sample_size_per_condition": expected_per_condition,
        "item_metrics": virtual_metrics,
        "summary": {"median_target_rho": float(np.nanmedian(target_rhos)) if any(value is not None for value in target_rhos) else None, "median_citc": float(np.nanmedian(citcs)) if any(value is not None for value in citcs) else None, "minimum_same_domain_vts": min((value for value in same_vts if value is not None), default=None), "minimum_cross_domain_vts": min((value for value in cross_vts if value is not None), default=None)},
        "interpretation": "固定三个顶层臂复用完全匹配的分数序列；每个facet group单独计算普通Spearman，臂级VTS取非目标组最大带符号rho，CITC过滤权仅属于target臂。",
    })
    recommendation_counts = {"retain": 0, "revise": 0, "remove": 0}
    for item in item_statistics.values():
        recommendation_counts[item["quality_evaluation"]["recommendation"]] += 1
    qualified_count = recommendation_counts["retain"]
    measurement_evaluation = _json_safe({
        "version": MEASUREMENT_EVALUATION_VERSION,
        "psychometric_analysis_round": analysis_round,
        "evidence_scope": "exploratory_virtual_screening_evidence",
        "overall_status": "development_ready" if qualified_count == len(item_order) else "development_ready_with_revision",
        "item_recommendation_counts": recommendation_counts,
        "reliability": {"overall_grade": "exploratory_virtual_screening", "cronbach_alpha": scale_statistics.get("cronbach_alpha"), "median_target_citc": validity_statistics["summary"]["median_citc"]},
        "validity": {"overall_grade": "exploratory_virtual_screening", "virtual_target_specificity": validity_statistics["summary"], "conditioning": {"variable": "condition_id", "method": "matched_facet_arms", "filtering_authority": True}},
        "interpretation": "四项资格门槛使用目标臂CITC、三臂rho和两类VTS；非目标facet group独立估计并取臂内最大带符号rho，目标组选项梯度仅为返修触发器。",
    })
    output_dir = manifest_path.parent / "psychometrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored_matched_condition_sjt_responses.csv"
    respondent_path = output_dir / "respondent_scores.csv"
    item_path = output_dir / "item_statistics.csv"
    quality_path = output_dir / "item_quality.csv"
    option_path = output_dir / "option_statistics.csv"
    option_json_path = output_dir / "option_choice_diagnostics.json"
    scale_path = output_dir / "scale_statistics.json"
    validity_path = output_dir / "virtual_screening_metrics.json"
    measurement_path = output_dir / "measurement_evaluation.json"
    analysis_path = output_dir / "analysis_manifest.json"
    _write_csv_atomic(scored_path, sjt_long)
    _write_csv_atomic(respondent_path, respondent_scores)
    _write_csv_atomic(item_path, item_frame)
    _write_csv_atomic(quality_path, item_quality_frame)
    _write_csv_atomic(option_path, option_frame)
    _write_json_atomic(option_json_path, {"schema_version": 3, "version": OPTION_CHOICE_DIAGNOSTICS_VERSION, "filtering_authority": False, "items": option_diagnostics})
    _write_json_atomic(scale_path, scale_statistics)
    _write_json_atomic(validity_path, validity_statistics)
    _write_json_atomic(measurement_path, measurement_evaluation)
    option_order_path = response_manifest.get("option_order_path")
    if not isinstance(option_order_path, str) or not Path(option_order_path).is_file():
        raise ValueError("作答 manifest 缺少有效 option_order_path")
    output_files = {"scored_matched_condition_sjt_responses": str(scored_path.resolve()), "respondent_scores": str(respondent_path.resolve()), "item_statistics": str(item_path.resolve()), "item_quality": str(quality_path.resolve()), "option_statistics": str(option_path.resolve()), "option_choice_diagnostics": str(option_json_path.resolve()), "scale_statistics": str(scale_path.resolve()), "virtual_screening_metrics": str(validity_path.resolve()), "measurement_evaluation": str(measurement_path.resolve()), "analysis_manifest": str(analysis_path.resolve())}
    analysis_manifest = {"schema_version": 6, "status": "completed", "formula_version": PSYCHOMETRIC_FORMULA_VERSION, "evaluation_version": MEASUREMENT_EVALUATION_VERSION, "psychometric_analysis_round": analysis_round, "run_id": state["run_id"], "item_bank_id": state["item_bank_id"], "item_bank_version": state["item_bank_version"], "item_bank_fingerprint": state.get("item_bank_fingerprint"), "sample_size": expected_respondents, "sample_size_per_condition": expected_per_condition, "condition_count": 3, "group_count": len(condition_ids), "condition_ids": list(condition_ids), "arm_ids": list(MATCHED_CONDITION_IDS), "sampling_design": "matched_facet_conditions", "conditions": deepcopy(conditions), "persona_mode": PERSONA_MODE_SCORE_PROFILE, "prompt_version": response_manifest.get("prompt_version"), "score_prompt_version": response_manifest.get("score_prompt_version"), "generator_version": response_manifest.get("generator_version"), "virtual_sample_config": deepcopy(response_manifest.get("virtual_sample_config") or {}), "item_count": len(item_order), "qualified_item_count": qualified_count, "qualification_rate": qualified_count / len(item_order), "criteria": {"iteration_gates": {"facet_citc_minimum": CITC_REVISION_THRESHOLD, "target_rho_minimum": TARGET_RHO_THRESHOLD, "same_domain_vts_minimum": SAME_DOMAIN_VTS_THRESHOLD, "cross_domain_vts_minimum": CROSS_DOMAIN_VTS_THRESHOLD, "condition_method": "fixed_three_arms_nested_facet_groups", "filtering_authority": True}, "target_option_gradient": {"filtering_authority": False, "repair_trigger": True}}, "formulas": {"facet_citc": "Pearson(target arm item, target arm same-facet remaining score sum)", "rho_target": "Spearman(Y_target, z_target)", "rho_group": "Spearman(Y_group, z_group), one group per non-target facet", "same_domain_vts": "rho_target - MAX(signed rho of same-domain facet groups)", "cross_domain_vts": "rho_target - MAX(signed rho of cross-domain facet groups)"}, "input_files": {"response_manifest": {"path": str(manifest_path.resolve()), "sha256": _file_sha256(manifest_path)}, "sjt_responses": {"path": str(sjt_path.resolve()), "sha256": _file_sha256(sjt_path)}, "score_profiles": {"path": str(Path(response_manifest["score_profiles_path"]).resolve()), "sha256": _file_sha256(Path(response_manifest["score_profiles_path"]))}, "option_orders": {"path": str(Path(option_order_path).resolve()), "sha256": _file_sha256(Path(option_order_path))}}, "output_files": output_files, "completed_at": utc_timestamp()}
    _write_json_atomic(analysis_path, analysis_manifest)
    test_statistics = {**scale_statistics, "sampling_design": "matched_facet_conditions", "condition_count": 3, "group_count": len(condition_ids), "condition_ids": list(condition_ids), "arm_ids": list(MATCHED_CONDITION_IDS), "sample_size_per_condition": expected_per_condition, "qualified_item_count": qualified_count, "qualification_rate": qualified_count / len(item_order), "qualification_criteria": analysis_manifest["criteria"], "virtual_screening_metrics": validity_statistics, "measurement_evaluation": measurement_evaluation, "output_files": output_files, "formula_version": PSYCHOMETRIC_FORMULA_VERSION, "evaluation_version": MEASUREMENT_EVALUATION_VERSION, "psychometric_analysis_round": analysis_round, "option_choice_diagnostics_version": OPTION_CHOICE_DIAGNOSTICS_VERSION}
    state_update = {"item_statistics": item_statistics, "test_statistics": _json_safe(test_statistics), "psychometric_analysis_round": analysis_round, "virtual_analysis_reconfiguration_reason": None}
    from sjt_system.evaluation.round_results import build_psychometric_round_result
    state_update["psychometric_round_result"] = build_psychometric_round_result({**state, **state_update})
    return {"state_update": state_update, "summary": f"已完成固定三臂、多 facet group 匹配分析：每组{expected_per_condition}人、共{len(condition_ids)}组、{len(item_order)}道题，保留{qualified_count}题，待返修{recommendation_counts['revise']}题。"}


def _option_frequency_group(
    item_responses: pd.DataFrame,
    *,
    item: Mapping[str, Any],
) -> dict[str, Any]:
    scoring_key = item.get("scoring_key") or {}
    option_metadata = {
        str(row.get("option_id")): row
        for row in item.get("response_options") or []
        if isinstance(row, Mapping) and row.get("option_id") is not None
    }
    counts = item_responses["selected_option_id"].astype(str).value_counts()
    group_n = int(len(item_responses))
    options: list[dict[str, Any]] = []
    for raw_option_id, raw_score in scoring_key.items():
        option_id = str(raw_option_id)
        count = int(counts.get(option_id, 0))
        metadata = option_metadata.get(option_id) or {}
        options.append(
            {
                "option_id": option_id,
                "behavioral_level": metadata.get("behavioral_level"),
                "score": float(raw_score),
                "selection_count": count,
                "selection_rate": (
                    float(count / group_n) if group_n > 0 else None
                ),
            }
        )
    return {"group_n": group_n, "options": options}


def _option_choice_diagnostics(
    *,
    sjt_long: pd.DataFrame,
    items: Mapping[str, Mapping[str, Any]],
    item_order: Sequence[str],
    score_frame: pd.DataFrame,
    score_specs: Sequence[Mapping[str, Any]],
    virtual_metrics: Mapping[str, Mapping[str, Any]],
    score_tiers: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Build non-filtering option frequencies for tiers and conditional bands."""

    spec_by_id = {
        str(spec.get("dimension_id")): dict(spec)
        for spec in score_specs
        if isinstance(spec, Mapping) and spec.get("dimension_id") is not None
    }
    tier_by_respondent = (
        sjt_long[["respondent_id", "tier_id"]]
        .drop_duplicates()
        .set_index("respondent_id")["tier_id"]
        .reindex(score_frame.index)
    )
    tier_scores = {
        str(row.get("tier_id")): row.get("facet_score")
        for row in score_tiers
        if isinstance(row, Mapping) and row.get("tier_id") is not None
    }
    diagnostics_by_item: dict[str, dict[str, Any]] = {}
    flat_rows: list[dict[str, Any]] = []

    def append_flat_rows(
        *,
        item_id: str,
        group: Mapping[str, Any],
        grouping_scope: str,
        tier_id: str | None = None,
        tier_facet_score: Any = None,
        dimension_role: str | None = None,
        vts_category: str | None = None,
        dimension_id: str | None = None,
        score_band: str | None = None,
        estimable: bool = True,
        contrasts: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        for option in group.get("options") or []:
            if not isinstance(option, Mapping):
                continue
            contrast = (contrasts or {}).get(str(option.get("option_id"))) or {}
            flat_rows.append(
                {
                    "item_id": item_id,
                    "grouping_scope": grouping_scope,
                    "tier_id": tier_id,
                    "tier_facet_score": tier_facet_score,
                    "dimension_role": dimension_role,
                    "vts_category": vts_category,
                    "dimension_id": dimension_id,
                    "score_band": score_band,
                    "group_n": group.get("group_n"),
                    "option_id": option.get("option_id"),
                    "behavioral_level": option.get("behavioral_level"),
                    "option_score": option.get("score"),
                    "selection_count": option.get("selection_count"),
                    "selection_rate": option.get("selection_rate"),
                    "high_low_rate_delta": contrast.get(
                        "high_low_rate_delta"
                    ),
                    "rate_range": contrast.get("rate_range"),
                    "estimable": bool(estimable),
                    "filtering_authority": False,
                }
            )

    for item_id in item_order:
        item = items[item_id]
        item_responses = sjt_long[sjt_long["item_id"] == item_id]
        aggregate = _option_frequency_group(item_responses, item=item)
        append_flat_rows(
            item_id=item_id,
            group=aggregate,
            grouping_scope="aggregate",
        )
        by_tier: list[dict[str, Any]] = []
        for tier_id in tier_scores:
            tier_group = _option_frequency_group(
                item_responses[item_responses["tier_id"] == tier_id],
                item=item,
            )
            tier_result = {
                "tier_id": tier_id,
                "facet_score": tier_scores[tier_id],
                **tier_group,
            }
            by_tier.append(tier_result)
            append_flat_rows(
                item_id=item_id,
                group=tier_result,
                grouping_scope="tier",
                tier_id=tier_id,
                tier_facet_score=tier_scores[tier_id],
            )

        specificity = virtual_metrics[item_id]["virtual_target_specificity"]
        target_id = str(item.get("target_dimension_id"))
        target_spec = spec_by_id.get(target_id) or {"dimension_id": target_id}
        dimension_rows: list[tuple[str, str | None, Mapping[str, Any]]] = [
            ("target", None, target_spec)
        ]
        for role, category, key in (
            (
                "same_domain_contaminant",
                "same_domain",
                "same_domain_non_target",
            ),
            (
                "cross_domain_contaminant",
                "cross_domain",
                "cross_domain_non_target",
            ),
        ):
            competitor = (specificity.get(key) or {}).get(
                "largest_non_target_facet"
            )
            if isinstance(competitor, Mapping) and competitor.get("dimension_id"):
                dimension_rows.append((role, category, competitor))

        conditional_rows: list[dict[str, Any]] = []
        seen_dimensions: set[tuple[str, str]] = set()
        for role, category, dimension in dimension_rows:
            dimension_id = str(dimension.get("dimension_id") or "")
            identity = (role, dimension_id)
            if not dimension_id or identity in seen_dimensions:
                continue
            seen_dimensions.add(identity)
            if dimension_id not in score_frame:
                band_assignments = pd.Series(index=score_frame.index, dtype="object")
                band_metadata = {
                    "conditioning_variable": "tier_id",
                    "method": "average_rank_residual_tertiles",
                    "rank_method": "average",
                    "tier_count": len(tier_scores),
                    "minimum_group_n": MINIMUM_OPTION_DIAGNOSTIC_GROUP_N,
                    "filtering_authority": False,
                    "estimable": False,
                    "reason": "分数档案缺少该构念分数",
                    "group_sizes": {"low": 0, "medium": 0, "high": 0},
                }
            else:
                band_assignments, band_metadata = (
                    _conditional_score_band_assignments(
                        score_frame[dimension_id], tier_by_respondent
                    )
                )
            groups: list[dict[str, Any]] = []
            rates_by_option: dict[str, dict[str, float | None]] = {}
            for band in ("low", "medium", "high"):
                respondent_ids = set(
                    band_assignments.index[band_assignments == band].astype(str)
                )
                band_group = _option_frequency_group(
                    item_responses[
                        item_responses["respondent_id"].isin(respondent_ids)
                    ],
                    item=item,
                )
                group_result = {"score_band": band, **band_group}
                groups.append(group_result)
                for option in band_group["options"]:
                    rates_by_option.setdefault(option["option_id"], {})[band] = (
                        option["selection_rate"]
                    )
            contrasts: list[dict[str, Any]] = []
            contrast_by_option: dict[str, dict[str, Any]] = {}
            for option_id, band_rates in rates_by_option.items():
                finite_rates = [
                    float(rate)
                    for rate in band_rates.values()
                    if rate is not None and math.isfinite(float(rate))
                ]
                low_rate = band_rates.get("low")
                high_rate = band_rates.get("high")
                contrast = {
                    "option_id": option_id,
                    "high_low_rate_delta": (
                        float(high_rate) - float(low_rate)
                        if high_rate is not None and low_rate is not None
                        else None
                    ),
                    "rate_range": (
                        max(finite_rates) - min(finite_rates)
                        if finite_rates else None
                    ),
                }
                contrasts.append(contrast)
                contrast_by_option[option_id] = contrast
            conditional_result = {
                "dimension_role": role,
                "vts_category": category,
                "dimension_id": dimension_id,
                "facet_name": dimension.get("facet_name"),
                "facet_name_en": dimension.get("facet_name_en"),
                "domain_id": dimension.get("domain_id"),
                "definition": dimension.get("definition"),
                "high_behavior": dimension.get("high_behavior"),
                "low_behavior": dimension.get("low_behavior"),
                **band_metadata,
                "groups": groups,
                "option_contrasts": contrasts,
            }
            conditional_rows.append(conditional_result)
            for group in groups:
                append_flat_rows(
                    item_id=item_id,
                    group=group,
                    grouping_scope="conditional_score_band",
                    dimension_role=role,
                    vts_category=category,
                    dimension_id=dimension_id,
                    score_band=str(group.get("score_band")),
                    estimable=bool(band_metadata.get("estimable")),
                    contrasts=contrast_by_option,
                )

        diagnostics_by_item[item_id] = _json_safe(
            {
                "version": OPTION_CHOICE_DIAGNOSTICS_VERSION,
                "filtering_authority": False,
                "interpretation": (
                    "选项频率仅用于定位可能需要文本核查的选项；"
                    "不能独立证明构念污染或触发返修。"
                ),
                "aggregate": aggregate,
                "by_tier": by_tier,
                "conditional_score_bands": conditional_rows,
            }
        )

    return diagnostics_by_item, pd.DataFrame(flat_rows)


    # 使用半量表均分；题目数为奇数时避免两半长度不同造成量尺差异。
    odd_score = frame[odd_columns].mean(axis=1)
def _scale_statistics(
    sjt_wide: pd.DataFrame,
    items: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    respondent_mean = sjt_wide.mean(axis=1)
    overall_alpha = cronbach_alpha(sjt_wide.to_numpy())
    dimensions: dict[str, Any] = {}
    dimension_ids = list(
        dict.fromkeys(
            str(item.get("target_dimension_id"))
            for item in items.values()
            if item.get("target_dimension_id") is not None
        )
    )
    for dimension_id in dimension_ids:
        columns = [
            item_id
            for item_id, item in items.items()
            if str(item.get("target_dimension_id")) == dimension_id
        ]
        frame = sjt_wide[columns]
        scores = frame.mean(axis=1)
        dimensions[dimension_id] = {
            "item_count": len(columns),
            "item_ids": columns,
            "mean": float(scores.mean()),
            "standard_deviation": float(scores.std(ddof=1)),
            "cronbach_alpha": cronbach_alpha(frame.to_numpy()),
        }
    return _json_safe(
        {
            "sample_size": len(sjt_wide),
            "item_count": len(sjt_wide.columns),
            "score_type": "mean_item_score",
            "score_minimum": float(respondent_mean.min()),
            "score_maximum": float(respondent_mean.max()),
            "score_mean": float(respondent_mean.mean()),
            "score_standard_deviation": float(
                respondent_mean.std(ddof=1)
            ),
            "cronbach_alpha": overall_alpha,
            "dimensions": dimensions,
        }
    )


def _respondent_scores(
    sjt_wide: pd.DataFrame,
    neo_dimensions: pd.DataFrame,
    items: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    output = pd.DataFrame(index=sjt_wide.index)
    output["sjt_total_sum"] = sjt_wide.sum(axis=1)
    output["sjt_total_mean"] = sjt_wide.mean(axis=1)
    dimension_ids = list(
        dict.fromkeys(
            str(item.get("target_dimension_id"))
            for item in items.values()
            if item.get("target_dimension_id") is not None
        )
    )
    for dimension_id in dimension_ids:
        columns = [
            item_id
            for item_id, item in items.items()
            if str(item.get("target_dimension_id")) == dimension_id
        ]
        output[f"sjt_dimension_{dimension_id}"] = sjt_wide[columns].mean(
            axis=1
        )
    for dimension_code in neo_dimensions.columns:
        output[f"neo_{dimension_code}"] = neo_dimensions[dimension_code]
    return output.reset_index()


def _validity_statistics(
    respondent_scores: pd.DataFrame,
    *,
    criterion: Mapping[str, str],
) -> dict[str, Any]:
    sjt_total = respondent_scores["sjt_total_mean"]
    neo_correlations = {
        dimension_code: _spearman(
            sjt_total,
            respondent_scores[f"neo_{dimension_code}"],
        )
        for dimension_code in ("N", "E", "O", "A", "C")
    }
    facet_correlations: dict[str, Any] = {}
    criterion_code = criterion["neo_ffi_dimension"]
    for column in respondent_scores.columns:
        if column.startswith("sjt_dimension_"):
            facet_correlations[column.removeprefix("sjt_dimension_")] = (
                _spearman(
                    respondent_scores[column],
                    respondent_scores[f"neo_{criterion_code}"],
                )
            )
    return _json_safe(
        {
            "method": "spearman",
            "criterion": dict(criterion),
            "target_convergent_pair": (
                f"sjt_total_mean ~ neo_{criterion_code}"
            ),
            "sjt_total_by_neo_dimension": neo_correlations,
            "sjt_dimension_by_target_neo_dimension": facet_correlations,
            "interpretation_limitations": [
                (
                    "SJT与Neo-FFI作答均由同一虚拟人格提示生成，"
                    "相关可能高估真实人类样本中的聚合效度。"
                ),
                (
                    "本结果是虚拟数据内部一致性证据，"
                    "不能替代真实被试效度研究。"
                ),
            ],
        }
    )


def _benjamini_hochberg(
    values: Sequence[float | None],
) -> list[float | None]:
    """Adjust a family of p-values while preserving missing positions."""

    values = list(values)
    valid = [
        (index, float(value))
        for index, value in enumerate(values)
        if value is not None and math.isfinite(float(value))
    ]
    adjusted: list[float | None] = [None] * len(values)
    if not valid:
        return adjusted
    ordered = sorted(valid, key=lambda pair: pair[1])
    running_minimum = 1.0
    count = len(ordered)
    for rank_from_end in range(count, 0, -1):
        index, p_value = ordered[rank_from_end - 1]
        candidate = min(1.0, p_value * count / rank_from_end)
        running_minimum = min(running_minimum, candidate)
        adjusted[index] = float(running_minimum)
    return adjusted


def _reliability_grade(value: float | None) -> str:
    if value is None:
        return "insufficient_evidence"
    if value >= 0.80:
        return "strong"
    if value >= 0.70:
        return "acceptable"
    if value >= 0.60:
        return "weak"
    return "inadequate"


def _enrich_item_quality(
    item_statistics: dict[str, dict[str, Any]],
    *,
    item_order: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Add facet-level discrimination and item-function decisions."""

    citc_adjusted = _benjamini_hochberg(
        item_statistics[item_id][
            "facet_corrected_item_total_correlation"
        ]["p_value"]
        for item_id in item_order
    )

    quality_rows: list[dict[str, Any]] = []
    for index, item_id in enumerate(item_order):
        item = item_statistics[item_id]
        item["facet_corrected_item_total_correlation"]["p_value_bh"] = (
            citc_adjusted[index]
        )

        citc = item["facet_corrected_item_total_correlation"]["r"]
        difficulty = item["difficulty"]
        option_statistics = item["option_statistics"]
        observed_option_count = sum(
            option["count"] > 0 for option in option_statistics.values()
        )
        effective_option_count = sum(
            option["rate"] >= MINIMUM_OPTION_RATE
            for option in option_statistics.values()
        )
        if citc is None:
            discrimination_rating = "insufficient_evidence"
        elif citc < 0:
            discrimination_rating = "poor"
        elif citc < CITC_REVISION_THRESHOLD:
            discrimination_rating = "weak"
        elif citc < CITC_MINIMUM_ACCEPTABLE:
            discrimination_rating = "acceptable_with_warning"
        elif citc >= CITC_STRONG:
            discrimination_rating = "strong"
        else:
            discrimination_rating = "acceptable"

        if 0.30 <= difficulty <= 0.75:
            distribution_rating = "strong"
        elif DIFFICULTY_LOWER_BOUND <= difficulty <= DIFFICULTY_UPPER_BOUND:
            distribution_rating = "acceptable"
        else:
            distribution_rating = "weak"

        if effective_option_count == len(option_statistics):
            option_rating = "strong"
        elif effective_option_count >= 3:
            option_rating = "acceptable_with_warning"
        else:
            option_rating = "weak"

        flags: list[str] = []
        if discrimination_rating == "poor":
            flags.append("区分方向错误或区分度过低")
        elif discrimination_rating == "weak":
            flags.append("区分度低于开发期建议值")
        elif discrimination_rating == "acceptable_with_warning":
            flags.append("分面内CITC为较弱支持性证据，不单独触发返修")
        if distribution_rating == "weak":
            flags.append("题目得分分布过于偏高或偏低")
        if option_rating == "weak":
            flags.append("实际仅有两个或更少选项发挥区分作用")
        elif option_rating == "acceptable_with_warning":
            flags.append("存在低频选项，需检查措辞而非自动删题")
        revision_needed = (
            discrimination_rating
            in {"poor", "weak", "insufficient_evidence"}
            or distribution_rating == "weak"
            or option_rating == "weak"
        )
        if revision_needed:
            recommendation = "revise"
            quality_grade = (
                "poor"
                if discrimination_rating == "poor"
                else "needs_revision"
            )
        else:
            recommendation = "retain"
            quality_grade = (
                "strong"
                if (
                    discrimination_rating == "strong"
                    and distribution_rating == "strong"
                    and option_rating == "strong"
                )
                else "acceptable"
            )

        evaluation = {
            "version": MEASUREMENT_EVALUATION_VERSION,
            "quality_grade": quality_grade,
            "recommendation": recommendation,
            "discrimination_rating": discrimination_rating,
            "distribution_rating": distribution_rating,
            "option_function_rating": option_rating,
            "observed_option_count": observed_option_count,
            "effective_option_count": effective_option_count,
            "diagnostic_flags": flags,
            "decision_rule": (
                "题项决策仅使用分面内CITC、标准化平均得分和"
                "有效选项数；单题与Neo-FFI上位ddomain的相关不参与返修。"
            ),
        }
        item["quality_evaluation"] = _json_safe(evaluation)
        quality_rows.append(
            {
                "item_id": item_id,
                "dimension_id": item.get("dimension_id"),
                "quality_grade": quality_grade,
                "recommendation": recommendation,
                "citc_r": citc,
                "citc_p_bh": citc_adjusted[index],
                "difficulty": difficulty,
                "observed_option_count": observed_option_count,
                "effective_option_count": effective_option_count,
                "minimum_option_rate": item["minimum_option_rate"],
                "diagnostic_flags": "；".join(flags),
            }
        )

    return item_statistics, pd.DataFrame(quality_rows)


def _apply_virtual_screening_quality(
    item_statistics: dict[str, dict[str, Any]],
    virtual_metrics: Mapping[str, Mapping[str, Any]],
    *,
    item_order: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Apply the four aggregate iteration gates; tier metrics stay descriptive."""

    rows: list[dict[str, Any]] = []
    for item_id in item_order:
        item = item_statistics[item_id]
        metrics = dict(virtual_metrics[item_id])
        citc = metrics["facet_citc"]
        specificity = metrics["virtual_target_specificity"]
        same_domain = specificity["same_domain_non_target"]
        cross_domain = specificity["cross_domain_non_target"]
        legacy_quality = deepcopy(item.get("quality_evaluation") or {})
        legacy_qualification = deepcopy(item.get("qualification") or {})
        flags: list[str] = []
        if citc.get("r") is None:
            flags.append("分面内CITC无法估计")
        elif not citc.get("passes"):
            flags.append("分面内CITC低于0.20")
        if specificity.get("target_spearman", {}).get("rho") is None:
            flags.append("控制tier_id后的目标秩相关无法估计")
        elif not specificity.get("target_rho_pass"):
            flags.append("控制tier_id后的目标相关低于0.30，可能存在特质激活不足")
        if same_domain.get("specificity_margin") is None:
            flags.append("同domain非目标facet VTS无法估计")
        elif not same_domain.get("passes"):
            flags.append("同domain非目标facet VTS低于0.10")
        if cross_domain.get("specificity_margin") is None:
            flags.append("不同domain非目标facet VTS无法估计")
        elif not cross_domain.get("passes"):
            flags.append("不同domain非目标facet VTS低于0.20")
        qualified = bool(metrics.get("passes"))
        recommendation = "retain" if qualified else "revise"
        quality = {
            "version": MEASUREMENT_EVALUATION_VERSION,
            "quality_grade": "acceptable" if qualified else "needs_revision",
            "recommendation": recommendation,
            "facet_citc": citc,
            "virtual_target_specificity": specificity,
            "per_tier_metrics": metrics.get("per_tier_metrics") or {},
            "diagnostic_flags": flags,
            "decision_rule": (
                "自动迭代使用分面内CITC>=.20、控制tier_id的条件目标"
                "Spearman rho>=.30、"
                "同domain非目标facet VTS>=.10和不同domain非目标facet VTS>=.20；"
                "难度、选项使用率和Cronbach alpha仅描述。"
            ),
            "descriptive_diagnostics": {
                "difficulty": item.get("difficulty"),
                "effective_option_count": legacy_quality.get(
                    "effective_option_count"
                ),
                "minimum_option_rate": item.get("minimum_option_rate"),
                "legacy_ratings": {
                    key: legacy_quality.get(key)
                    for key in (
                        "discrimination_rating",
                        "distribution_rating",
                        "option_function_rating",
                    )
                },
                "legacy_qualification": legacy_qualification,
            },
            # Compatibility fields for report/sort code; they are descriptive.
            "discrimination_rating": legacy_quality.get(
                "discrimination_rating"
            ),
            "distribution_rating": legacy_quality.get("distribution_rating"),
            "option_function_rating": legacy_quality.get(
                "option_function_rating"
            ),
            "observed_option_count": legacy_quality.get(
                "observed_option_count"
            ),
            "effective_option_count": legacy_quality.get(
                "effective_option_count"
            ),
        }
        item["virtual_screening_metrics"] = _json_safe(metrics)
        item["qualification"] = {
            "citc_pass": bool(citc.get("passes")),
            "target_rho_pass": bool(specificity.get("target_rho_pass")),
            "same_domain_vts_pass": bool(same_domain.get("passes")),
            "cross_domain_vts_pass": bool(cross_domain.get("passes")),
            "qualified": qualified,
        }
        item["quality_evaluation"] = _json_safe(quality)
        rows.append(
            {
                "item_id": item_id,
                "metric_scope": "aggregate",
                "tier_id": None,
                "filtering_authority": True,
                "correlation_method": "rank_residualization",
                "conditioning_variable": "tier_id",
                "dimension_id": item.get("dimension_id"),
                "quality_grade": quality["quality_grade"],
                "recommendation": recommendation,
                "citc_r": citc.get("r"),
                "target_rho": specificity.get("target_spearman", {}).get(
                    "rho"
                ),
                "conditional_target_rho": specificity.get(
                    "conditional_target_spearman", {}
                ).get("rho"),
                "same_domain_largest_non_target_dimension_id": same_domain.get(
                    "largest_non_target_dimension_id"
                ),
                "same_domain_largest_non_target_conditional_rho": same_domain.get(
                    "largest_non_target_conditional_rho"
                ),
                "same_domain_max_non_target_rho": same_domain.get(
                    "max_non_target_rho"
                ),
                "same_domain_vts": same_domain.get("specificity_margin"),
                "cross_domain_largest_non_target_dimension_id": cross_domain.get(
                    "largest_non_target_dimension_id"
                ),
                "cross_domain_largest_non_target_conditional_rho": cross_domain.get(
                    "largest_non_target_conditional_rho"
                ),
                "cross_domain_max_non_target_rho": cross_domain.get(
                    "max_non_target_rho"
                ),
                "cross_domain_vts": cross_domain.get("specificity_margin"),
                "difficulty_descriptive": item.get("difficulty"),
                "effective_option_count_descriptive": legacy_quality.get(
                    "effective_option_count"
                ),
                "diagnostic_flags": "；".join(flags),
            }
        )
        for tier_id, tier_metric in (metrics.get("per_tier_metrics") or {}).items():
            tier_citc = (tier_metric or {}).get("facet_citc") or {}
            tier_specificity = (
                (tier_metric or {}).get("virtual_target_specificity") or {}
            )
            tier_same = tier_specificity.get("same_domain_non_target") or {}
            tier_cross = tier_specificity.get("cross_domain_non_target") or {}
            rows.append(
                {
                    "item_id": item_id,
                    "metric_scope": "tier",
                    "tier_id": tier_id,
                    "filtering_authority": False,
                    "correlation_method": "ordinary_spearman",
                    "conditioning_variable": None,
                    "dimension_id": item.get("dimension_id"),
                    "quality_grade": "diagnostic_only",
                    "recommendation": "diagnostic_only",
                    "citc_r": tier_citc.get("r"),
                    "target_rho": (
                        tier_specificity.get("target_spearman") or {}
                    ).get("rho"),
                    "conditional_target_rho": None,
                    "ordinary_target_rho": (
                        tier_specificity.get("target_spearman") or {}
                    ).get("rho"),
                    "same_domain_largest_non_target_dimension_id": tier_same.get(
                        "largest_non_target_dimension_id"
                    ),
                    "same_domain_max_non_target_rho": tier_same.get(
                        "max_non_target_rho"
                    ),
                    "same_domain_largest_non_target_conditional_rho": None,
                    "same_domain_vts": tier_same.get("specificity_margin"),
                    "cross_domain_largest_non_target_dimension_id": tier_cross.get(
                        "largest_non_target_dimension_id"
                    ),
                    "cross_domain_max_non_target_rho": tier_cross.get(
                        "max_non_target_rho"
                    ),
                    "cross_domain_largest_non_target_conditional_rho": None,
                    "cross_domain_vts": tier_cross.get("specificity_margin"),
                    "difficulty_descriptive": None,
                    "effective_option_count_descriptive": None,
                    "diagnostic_flags": "各分数档指标不参与过滤",
                }
            )
    return item_statistics, pd.DataFrame(rows)


def _evaluate_test_quality(
    *,
    item_statistics: Mapping[str, Mapping[str, Any]],
    scale_statistics: Mapping[str, Any],
    validity_statistics: Mapping[str, Any],
    criterion: Mapping[str, str],
) -> dict[str, Any]:
    alpha = scale_statistics.get("cronbach_alpha")
    alpha_grade = _reliability_grade(alpha)
    reliability_grade = alpha_grade

    correlations = validity_statistics.get(
        "sjt_total_by_neo_dimension", {}
    )
    criterion_code = criterion["neo_ffi_dimension"]
    target = correlations.get(criterion_code, {})
    target_rho = target.get("rho")
    non_target = {
        code: result.get("rho")
        for code, result in correlations.items()
        if code != criterion_code and result.get("rho") is not None
    }
    max_non_target_code = None
    max_non_target_rho = None
    if non_target:
        max_non_target_code, max_non_target_rho = max(
            non_target.items(),
            key=lambda pair: abs(pair[1]),
        )
    discriminant_margin = (
        abs(target_rho) - abs(max_non_target_rho)
        if target_rho is not None and max_non_target_rho is not None
        else None
    )
    if target_rho is None:
        convergent_grade = "insufficient_evidence"
    elif abs(target_rho) >= 0.50:
        convergent_grade = "strong_numeric_association"
    elif abs(target_rho) >= 0.30:
        convergent_grade = "moderate_numeric_association"
    else:
        convergent_grade = "weak_numeric_association"

    if (
        target_rho is None
        or max_non_target_rho is None
        or discriminant_margin is None
    ):
        discriminant_grade = "insufficient_evidence"
    elif (
        abs(max_non_target_rho) < 0.30
        and discriminant_margin >= 0.20
    ):
        discriminant_grade = "supportive"
    elif discriminant_margin >= 0.20:
        discriminant_grade = "mixed"
    else:
        discriminant_grade = "weak"

    recommendation_counts: dict[str, int] = {}
    for item in item_statistics.values():
        recommendation = (
            item.get("quality_evaluation", {}).get("recommendation")
        )
        if recommendation:
            recommendation_counts[recommendation] = (
                recommendation_counts.get(recommendation, 0) + 1
            )

    missing_evidence = [
        "独立真实被试的聚合与区分效度",
        "预测效度或效标关联效度",
        "重测信度",
    ]
    overall_status = (
        "development_ready_with_revision"
        if reliability_grade in {"strong", "acceptable"}
        and target_rho is not None
        and target_rho > 0
        else "not_ready"
    )

    return _json_safe(
        {
            "version": MEASUREMENT_EVALUATION_VERSION,
            "evidence_scope": "virtual_development_sample",
            "criterion": dict(criterion),
            "overall_status": overall_status,
            "operational_use_status": "not_validated_for_operational_use",
            "item_recommendation_counts": recommendation_counts,
            "reliability": {
                "overall_grade": reliability_grade,
                "cronbach_alpha": alpha,
                "cronbach_alpha_grade": alpha_grade,
                "interpretation": (
                    "整体内部一致性按开发期标准评价；分维度仅2至3题，"
                    "其alpha不作为独立分量表定论。"
                ),
            },
            "validity": {
                "overall_grade": "preliminary_incomplete",
                "convergent": {
                    "grade": convergent_grade,
                    "target": f"Neo-FFI {criterion_code}",
                    "rho": target_rho,
                    "p_value": target.get("p_value"),
                    "limitation": (
                        "SJT与Neo-FFI由同一虚拟人格提示生成，"
                        "该相关不能当作独立效度证据。"
                    ),
                },
                "discriminant": {
                    "grade": discriminant_grade,
                    "largest_non_target_dimension": max_non_target_code,
                    "largest_non_target_rho": max_non_target_rho,
                    "target_margin": discriminant_margin,
                },
            },
            "missing_evidence": missing_evidence,
            "interpretation": (
                "当前结果可以支持开发期保留、修改和补测决策，"
                "不能支持正式选拔、诊断或个体高风险决策。"
            ),
        }
    )


def _manifest_persona_modes(
    manifest: Mapping[str, Any],
    sjt_records: Sequence[Mapping[str, Any]],
) -> list[str]:
    raw_modes = manifest.get("persona_modes")
    if raw_modes is None:
        inferred = sorted(
            {
                str(record["persona_mode"])
                for record in sjt_records
                if isinstance(record.get("persona_mode"), str)
            }
        )
        return inferred or [PERSONA_MODE_SUMMARY_PLUS_ITEMS]
    if (
        not isinstance(raw_modes, list)
        or not raw_modes
        or any(not isinstance(mode, str) or not mode for mode in raw_modes)
        or len(set(raw_modes)) != len(raw_modes)
    ):
        raise ValueError("response manifest 包含无效 persona_modes")
    if any(mode != PERSONA_MODE_SUMMARY_PLUS_ITEMS for mode in raw_modes):
        raise ValueError("response manifest 只支持 summary_plus_items")
    return list(raw_modes)


def _filter_persona_mode_records(
    records: Sequence[Mapping[str, Any]],
    *,
    persona_mode: str,
) -> list[dict[str, Any]]:
    records_with_mode = [
        record for record in records if "persona_mode" in record
    ]
    if not records_with_mode:
        return [dict(record) for record in records]
    if len(records_with_mode) != len(records):
        raise ValueError("同一作答文件不能混合有无 persona_mode 的记录")
    filtered = [
        dict(record)
        for record in records
        if record.get("persona_mode") == persona_mode
    ]
    if not filtered:
        raise ValueError(f"缺少 persona_mode={persona_mode} 的作答")
    return filtered


def _paired_choice_comparison(
    records: Sequence[Mapping[str, Any]],
    *,
    primary_mode: str,
    comparison_mode: str,
) -> dict[str, Any]:
    by_mode: dict[str, dict[tuple[str, str], str]] = {}
    for mode in (primary_mode, comparison_mode):
        keyed: dict[tuple[str, str], str] = {}
        for record in records:
            if record.get("persona_mode") != mode:
                continue
            key = (
                str(record.get("respondent_id")),
                str(record.get("item_id")),
            )
            if key in keyed:
                raise ValueError(
                    f"{mode} SJT 作答包含重复 respondent-item 记录"
                )
            keyed[key] = str(record.get("selected_option_id"))
        by_mode[mode] = keyed
    primary = by_mode[primary_mode]
    comparison = by_mode[comparison_mode]
    if set(primary) != set(comparison):
        raise ValueError("两种 persona_mode 的 SJT 配对记录集合不一致")
    item_totals: dict[str, int] = {}
    item_matches: dict[str, int] = {}
    matches = 0
    for key, primary_choice in primary.items():
        item_id = key[1]
        item_totals[item_id] = item_totals.get(item_id, 0) + 1
        matched = primary_choice == comparison[key]
        matches += int(matched)
        item_matches[item_id] = item_matches.get(item_id, 0) + int(matched)
    total = len(primary)
    return {
        "primary_mode": primary_mode,
        "comparison_mode": comparison_mode,
        "paired_response_count": total,
        "overall_option_agreement": matches / total if total else None,
        "item_option_agreement": {
            item_id: item_matches.get(item_id, 0) / count
            for item_id, count in item_totals.items()
        },
    }


def _run_paired_persona_mode_analysis(
    *,
    state: PSJTState,
    manifest_path: Path,
    response_manifest: Mapping[str, Any],
    sjt_records: Sequence[Mapping[str, Any]],
    neo_records: Sequence[Mapping[str, Any]],
    persona_modes: Sequence[str],
) -> dict[str, Any]:
    """Analyze each paired prompt condition independently."""

    primary_mode = persona_modes[0]
    mode_root = manifest_path.parent / "psychometrics" / "persona_modes"
    mode_results: dict[str, dict[str, Any]] = {}
    for persona_mode in persona_modes:
        input_dir = mode_root / persona_mode / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        mode_sjt_path = input_dir / "sjt_responses.jsonl"
        mode_neo_path = input_dir / "neo_ffi_responses.jsonl"
        mode_manifest_path = input_dir / "manifest.json"
        mode_sjt_records = _filter_persona_mode_records(
            sjt_records,
            persona_mode=persona_mode,
        )
        mode_neo_records = _filter_persona_mode_records(
            neo_records,
            persona_mode=persona_mode,
        )
        _write_jsonl_atomic(mode_sjt_path, mode_sjt_records)
        _write_jsonl_atomic(mode_neo_path, mode_neo_records)
        mode_manifest = {
            **dict(response_manifest),
            "persona_modes": [persona_mode],
            "persona_mode_count": 1,
            "expected_sjt_records": len(mode_sjt_records),
            "completed_sjt_records": len(mode_sjt_records),
            "expected_neo_ffi_records": len(mode_neo_records),
            "completed_neo_ffi_records": len(mode_neo_records),
            "parent_response_manifest": str(manifest_path),
        }
        _write_json_atomic(mode_manifest_path, mode_manifest)
        mode_state = dict(state)
        mode_state["virtual_response_data_ref"] = str(mode_manifest_path)
        result = run_psychometric_analysis(mode_state)  # one-mode base case
        mode_results[persona_mode] = result

    comparisons = {
        persona_mode: _paired_choice_comparison(
            sjt_records,
            primary_mode=primary_mode,
            comparison_mode=persona_mode,
        )
        for persona_mode in persona_modes
        if persona_mode != primary_mode
    }
    comparison_path = (
        manifest_path.parent
        / "psychometrics"
        / "paired_persona_mode_comparison.json"
    )
    comparison_payload = {
        "schema_version": 1,
        "primary_mode": primary_mode,
        "persona_modes": list(persona_modes),
        "comparisons": comparisons,
        "interpretation": (
            "The same respondents completed the same items under each prompt "
            "condition. Differences estimate prompt-conditioning effects, not "
            "independent-sample effects."
        ),
    }
    _write_json_atomic(comparison_path, comparison_payload)

    primary = mode_results[primary_mode]
    primary_update = dict(primary["state_update"])
    primary_test_statistics = dict(primary_update["test_statistics"])
    primary_test_statistics.update(
        {
            "primary_persona_mode": primary_mode,
            "persona_modes": list(persona_modes),
            "persona_mode_results": {
                mode: {
                    "item_statistics": result["state_update"][
                        "item_statistics"
                    ],
                    "test_statistics": result["state_update"][
                        "test_statistics"
                    ],
                }
                for mode, result in mode_results.items()
            },
            "paired_persona_mode_comparison": comparison_payload,
            "paired_persona_mode_comparison_path": str(
                comparison_path.resolve()
            ),
        }
    )
    primary_update["test_statistics"] = _json_safe(
        primary_test_statistics
    )
    return {
        "state_update": primary_update,
        "summary": (
            f"已分别完成 {len(persona_modes)} 种人格提示模式的心理测量分析；"
            f"下游题目筛选以 {primary_mode} 为主分析，其他模式用于配对敏感性比较。"
        ),
    }


def _run_score_profile_analysis(
    *,
    state: PSJTState,
    manifest_path: Path,
    sjt_path: Path,
    response_manifest: Mapping[str, Any],
    item_order: list[str],
    items: Mapping[str, Mapping[str, Any]],
    expected_respondents: int,
) -> dict[str, Any]:
    """Analyze tiered single responses with four aggregate iteration gates."""

    persona_modes = response_manifest.get("persona_modes")
    if persona_modes != [PERSONA_MODE_SCORE_PROFILE]:
        raise ValueError("schema_version=3 的作答必须使用 score_profile")
    score_tiers = response_manifest.get("score_tiers")
    if not isinstance(score_tiers, list) or not 1 <= len(score_tiers) <= 3:
        raise ValueError("分档分析要求1到3个 score_tiers")
    tier_ids = [str(tier.get("tier_id")) for tier in score_tiers if isinstance(tier, Mapping)]
    if len(tier_ids) != len(score_tiers) or len(set(tier_ids)) != len(tier_ids):
        raise ValueError("score_tiers 缺少唯一 tier_id")
    sjt_records = _filter_persona_mode_records(
        _read_jsonl(sjt_path), persona_mode=PERSONA_MODE_SCORE_PROFILE
    )
    sjt_long, sjt_wide, tier_wide = _score_tiered_sjt(
        sjt_records,
        item_order=item_order,
        items=items,
        expected_respondent_count=expected_respondents,
        expected_tier_ids=tier_ids,
    )
    score_profiles_path = response_manifest.get("score_profiles_path")
    if not isinstance(score_profiles_path, str):
        raise ValueError("作答 manifest 缺少 score_profiles_path")
    score_payload = _read_json(Path(score_profiles_path))
    score_specs = score_payload.get("score_specs")
    if not isinstance(score_specs, list) or not score_specs:
        raise ValueError("score_profiles.json 缺少 score_specs")
    score_frame = _score_profile_frame(
        score_payload,
        expected_respondents=list(sjt_wide.index),
        score_specs=score_specs,
    )
    virtual_metrics = _virtual_item_metrics(
        item_order=item_order,
        items=items,
        response_wide=sjt_wide,
        tier_wide=tier_wide,
        score_frame=score_frame,
        score_specs=score_specs,
    )
    item_statistics, item_frame, option_frame = _item_statistics(
        sjt_long,
        sjt_wide,
        item_order=item_order,
        items=items,
    )
    item_statistics, _ = _enrich_item_quality(
        item_statistics,
        item_order=item_order,
    )
    item_statistics, item_quality_frame = _apply_virtual_screening_quality(
        item_statistics,
        virtual_metrics,
        item_order=item_order,
    )
    option_diagnostics, option_frame = _option_choice_diagnostics(
        sjt_long=sjt_long,
        items=items,
        item_order=item_order,
        score_frame=score_frame,
        score_specs=score_specs,
        virtual_metrics=virtual_metrics,
        score_tiers=score_tiers,
    )
    for item_id in item_order:
        item_statistics[item_id]["option_choice_diagnostics"] = deepcopy(
            option_diagnostics[item_id]
        )
    scale_statistics = _scale_statistics(sjt_wide, items)
    analysis_round = int(state.get("psychometric_analysis_round") or 0) + 1

    respondent_scores = score_frame.copy()
    respondent_scores.insert(0, "respondent_id", respondent_scores.index)
    tier_by_respondent = (
        sjt_long[["respondent_id", "tier_id"]]
        .drop_duplicates()
        .set_index("respondent_id")["tier_id"]
    )
    respondent_scores.insert(
        1, "tier_id", [tier_by_respondent.loc[index] for index in respondent_scores.index]
    )
    respondent_scores["sjt_total_score"] = sjt_wide.mean(axis=1).values
    for dimension_id in list(
        dict.fromkeys(str(items[item_id]["target_dimension_id"]) for item_id in item_order)
    ):
        columns = [
            item_id
            for item_id in item_order
            if str(items[item_id]["target_dimension_id"]) == dimension_id
        ]
        respondent_scores[f"sjt_{dimension_id}"] = sjt_wide[columns].mean(
            axis=1
        ).values
    respondent_scores = respondent_scores.reset_index(drop=True)

    target_rhos = [
        metrics["virtual_target_specificity"]["target_spearman"]["rho"]
        for metrics in virtual_metrics.values()
        if metrics["virtual_target_specificity"]["target_spearman"]["rho"]
        is not None
    ]
    citc_values = [
        metrics["facet_citc"]["r"]
        for metrics in virtual_metrics.values()
        if metrics["facet_citc"]["r"] is not None
    ]
    same_domain_margins = [
        metrics["virtual_target_specificity"]["same_domain_non_target"]["specificity_margin"]
        for metrics in virtual_metrics.values()
        if metrics["virtual_target_specificity"]["same_domain_non_target"]["specificity_margin"] is not None
    ]
    cross_domain_margins = [
        metrics["virtual_target_specificity"]["cross_domain_non_target"]["specificity_margin"]
        for metrics in virtual_metrics.values()
        if metrics["virtual_target_specificity"]["cross_domain_non_target"]["specificity_margin"] is not None
    ]
    validity_statistics = _json_safe(
        {
            "evidence_scope": "exploratory_virtual_score_manipulation",
            "psychometric_analysis_round": analysis_round,
            "item_metrics": virtual_metrics,
            "summary": {
                "median_target_rho": (
                    float(np.median(target_rhos)) if target_rhos else None
                ),
                "median_citc": (
                    float(np.median(citc_values)) if citc_values else None
                ),
                "minimum_same_domain_vts": (
                    min(same_domain_margins) if same_domain_margins else None
                ),
                "minimum_cross_domain_vts": (
                    min(cross_domain_margins) if cross_domain_margins else None
                ),
                "tier_count": len(tier_ids),
            },
            "interpretation": (
                "总目标相关与VTS均先对平均秩控制tier_id；各档普通相关仅作诊断。"
                "分数操纵与同一模型作答形成探索性虚拟目标特异性证据；"
                "不能替代SME内容效度和真人构念/效标效度。"
            ),
        }
    )
    recommendation_counts = {"retain": 0, "revise": 0, "remove": 0}
    for item in item_statistics.values():
        recommendation = item["quality_evaluation"]["recommendation"]
        recommendation_counts[recommendation] += 1
    qualified_count = recommendation_counts["retain"]
    measurement_evaluation = _json_safe(
        {
            "version": MEASUREMENT_EVALUATION_VERSION,
            "psychometric_analysis_round": analysis_round,
            "evidence_scope": "exploratory_virtual_screening_evidence",
            "overall_status": (
                "development_ready"
                if qualified_count == len(item_order)
                else "development_ready_with_revision"
            ),
            "operational_use_status": "not_validated_for_operational_use",
            "item_recommendation_counts": recommendation_counts,
            "reliability": {
                "overall_grade": "exploratory_virtual_screening",
                "cronbach_alpha": scale_statistics.get("cronbach_alpha"),
                "median_facet_citc": validity_statistics["summary"]["median_citc"],
                "interpretation": (
                    "单题迭代一致性门槛是合并分数档后的分面内CITC>=.20；"
                    "Cronbach alpha仅作描述。"
                ),
            },
            "validity": {
                "overall_grade": "exploratory_virtual_screening",
                "virtual_target_specificity": validity_statistics["summary"],
                "conditioning": {
                    "variable": "tier_id",
                    "method": "rank_residualization",
                    "rank_method": "average",
                },
                "limitation": (
                    "显式人格分数与SJT反应由同一提示和模型连接，不能称为"
                    "独立正式效度。"
                ),
            },
            "missing_evidence": [
                "独立SME盲评内容效度",
                "真人样本的结构、收敛、区分与效标效度",
                "真人重测信度",
            ],
            "interpretation": (
                "自动返修仅由总CITC、条件目标相关及拆分后的两类条件VTS触发；"
                "所有结论限于探索性虚拟预筛。"
            ),
        }
    )

    output_dir = manifest_path.parent / "psychometrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_sjt_path = output_dir / "scored_tiered_sjt_responses.csv"
    respondent_scores_path = output_dir / "respondent_scores.csv"
    item_statistics_path = output_dir / "item_statistics.csv"
    item_quality_path = output_dir / "item_quality.csv"
    option_statistics_path = output_dir / "option_statistics.csv"
    option_diagnostics_path = output_dir / "option_choice_diagnostics.json"
    scale_statistics_path = output_dir / "scale_statistics.json"
    validity_statistics_path = output_dir / "virtual_screening_metrics.json"
    measurement_evaluation_path = output_dir / "measurement_evaluation.json"
    analysis_manifest_path = output_dir / "analysis_manifest.json"
    _write_csv_atomic(scored_sjt_path, sjt_long)
    _write_csv_atomic(respondent_scores_path, respondent_scores)
    _write_csv_atomic(item_statistics_path, item_frame)
    _write_csv_atomic(item_quality_path, item_quality_frame)
    _write_csv_atomic(option_statistics_path, option_frame)
    _write_json_atomic(
        option_diagnostics_path,
        {
            "schema_version": 1,
            "version": OPTION_CHOICE_DIAGNOSTICS_VERSION,
            "filtering_authority": False,
            "minimum_group_n": MINIMUM_OPTION_DIAGNOSTIC_GROUP_N,
            "psychometric_analysis_round": analysis_round,
            "items": option_diagnostics,
        },
    )
    _write_json_atomic(scale_statistics_path, scale_statistics)
    _write_json_atomic(validity_statistics_path, validity_statistics)
    _write_json_atomic(measurement_evaluation_path, measurement_evaluation)

    warnings = [
        "虚拟指标仅用于开发期预筛，不能作为正式单题信效度结论。"
    ]
    if expected_respondents < 100:
        warnings.append("总样本量少于100，CITC与相关系数可能有较大波动。")
    output_files = {
        "scored_tiered_sjt_responses": str(scored_sjt_path.resolve()),
        "respondent_scores": str(respondent_scores_path.resolve()),
        "item_statistics": str(item_statistics_path.resolve()),
        "item_quality": str(item_quality_path.resolve()),
        "option_statistics": str(option_statistics_path.resolve()),
        "option_choice_diagnostics": str(option_diagnostics_path.resolve()),
        "scale_statistics": str(scale_statistics_path.resolve()),
        "virtual_screening_metrics": str(validity_statistics_path.resolve()),
        "measurement_evaluation": str(
            measurement_evaluation_path.resolve()
        ),
        "analysis_manifest": str(analysis_manifest_path.resolve()),
    }
    option_order_path = response_manifest.get("option_order_path")
    if not isinstance(option_order_path, str) or not Path(
        option_order_path
    ).is_file():
        raise ValueError("作答 manifest 缺少有效 option_order_path")
    analysis_manifest = {
        "schema_version": 5,
        "status": "completed",
        "formula_version": PSYCHOMETRIC_FORMULA_VERSION,
        "evaluation_version": MEASUREMENT_EVALUATION_VERSION,
        "psychometric_analysis_round": analysis_round,
        "run_id": state["run_id"],
        "item_bank_id": state["item_bank_id"],
        "item_bank_version": state["item_bank_version"],
        "item_bank_fingerprint": state["item_bank_fingerprint"],
        "sample_size": expected_respondents,
        "tier_count": len(tier_ids),
        "score_tiers": deepcopy(score_tiers),
        "persona_mode": PERSONA_MODE_SCORE_PROFILE,
        "prompt_version": response_manifest.get("prompt_version"),
        "score_prompt_version": response_manifest.get(
            "score_prompt_version"
        ),
        "generator_version": response_manifest.get("generator_version"),
        "virtual_sample_config": deepcopy(
            response_manifest.get("virtual_sample_config") or {}
        ),
        "item_count": len(item_order),
        "qualified_item_count": qualified_count,
        "qualification_rate": qualified_count / len(item_order),
        "criteria": {
            "iteration_gates": {
                "facet_citc_minimum": CITC_REVISION_THRESHOLD,
                "conditional_target_spearman_rho_minimum": TARGET_RHO_THRESHOLD,
                "same_domain_vts_minimum": SAME_DOMAIN_VTS_THRESHOLD,
                "cross_domain_vts_minimum": CROSS_DOMAIN_VTS_THRESHOLD,
                "conditioning_variable": "tier_id",
                "conditioning_method": "rank_residualization",
            },
            "descriptive_only": {
                "difficulty_range": [DIFFICULTY_LOWER_BOUND, DIFFICULTY_UPPER_BOUND],
                "minimum_option_rate": MINIMUM_OPTION_RATE,
                "cronbach_alpha": True,
                "option_choice_diagnostics": {
                    "version": OPTION_CHOICE_DIAGNOSTICS_VERSION,
                    "minimum_group_n": MINIMUM_OPTION_DIAGNOSTIC_GROUP_N,
                    "filtering_authority": False,
                },
            },
        },
        "formulas": {
            "facet_citc": (
                "Pearson(item score, sum of other items with the same target facet)"
            ),
            "same_domain_vts": (
                "conditional Spearman(target score, item score | tier_id) - maximum "
                "absolute conditional Spearman(same-domain non-target facet score, "
                "item score | tier_id)"
            ),
            "cross_domain_vts": (
                "conditional Spearman(target score, item score | tier_id) - maximum "
                "absolute conditional Spearman(cross-domain non-target facet score, "
                "item score | tier_id)"
            ),
            "difficulty_descriptive": (
                "(mean item score - theoretical min) / range"
            ),
        },
        "input_files": {
            "response_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": _file_sha256(manifest_path),
            },
            "sjt_responses": {
                "path": str(sjt_path.resolve()),
                "sha256": _file_sha256(sjt_path),
            },
            "score_profiles": {
                "path": str(Path(score_profiles_path).resolve()),
                "sha256": _file_sha256(Path(score_profiles_path)),
            },
            "option_orders": {
                "path": str(Path(option_order_path).resolve()),
                "sha256": _file_sha256(Path(option_order_path)),
            },
        },
        "output_files": output_files,
        "warnings": warnings,
        "completed_at": utc_timestamp(),
    }
    _write_json_atomic(analysis_manifest_path, analysis_manifest)
    test_statistics = {
        **scale_statistics,
        "persona_mode": PERSONA_MODE_SCORE_PROFILE,
        "tier_count": len(tier_ids),
        "score_tiers": deepcopy(score_tiers),
        "qualified_item_count": qualified_count,
        "qualification_rate": qualified_count / len(item_order),
        "qualification_criteria": analysis_manifest["criteria"],
        "virtual_screening_metrics": validity_statistics,
        "measurement_evaluation": measurement_evaluation,
        "warnings": warnings,
        "output_files": output_files,
        "formula_version": PSYCHOMETRIC_FORMULA_VERSION,
        "evaluation_version": MEASUREMENT_EVALUATION_VERSION,
        "psychometric_analysis_round": analysis_round,
        "option_choice_diagnostics_version": OPTION_CHOICE_DIAGNOSTICS_VERSION,
    }
    state_update = {
        "item_statistics": item_statistics,
        "test_statistics": _json_safe(test_statistics),
        "psychometric_analysis_round": analysis_round,
        "virtual_analysis_reconfiguration_reason": None,
    }
    from sjt_system.evaluation.round_results import (
        build_psychometric_round_result,
    )

    state_update["psychometric_round_result"] = (
        build_psychometric_round_result({**state, **state_update})
    )
    return {
        "state_update": state_update,
        "summary": (
            f"已完成 {expected_respondents} 名分数型虚拟被试、"
            f"{len(tier_ids)} 个分数档、{len(item_order)} 道题的四门槛分析；"
            f"保留 {recommendation_counts['retain']} 题，"
            f"修改 {recommendation_counts['revise']} 题。"
        ),
    }


def run_psychometric_analysis(
    state: PSJTState,
) -> dict[str, Any]:
    """Score saved responses, compute paper-aligned CTT metrics, and persist them."""

    manifest_path, sjt_path, neo_path, response_manifest = (
        _resolve_response_paths(state)
    )
    item_order, items = _prepare_item_contract(state)
    expected_respondents = response_manifest.get("sample_size")
    if (
        not isinstance(expected_respondents, int)
        or isinstance(expected_respondents, bool)
        or expected_respondents < 1
    ):
        raise ValueError("虚拟作答 manifest 缺少有效 sample_size")
    if response_manifest.get("schema_version") == MATCHED_CONDITION_SCHEMA_VERSION:
        return _run_matched_condition_analysis(
            state=state,
            manifest_path=manifest_path,
            sjt_path=sjt_path,
            response_manifest=response_manifest,
            item_order=item_order,
            items=items,
            expected_respondents=expected_respondents,
        )
    raise ValueError(
        "旧 tier/重复作答/条件残差化虚拟作答结果已失效；"
        f"请使用 schema_version={MATCHED_CONDITION_SCHEMA_VERSION} 的固定三臂、多 facet group 匹配协议重新作答"
    )

    sjt_records = _read_jsonl(sjt_path)
    neo_records = _read_jsonl(neo_path)
    persona_modes = _manifest_persona_modes(
        response_manifest,
        sjt_records,
    )
    if len(persona_modes) > 1:
        return _run_paired_persona_mode_analysis(
            state=state,
            manifest_path=manifest_path,
            response_manifest=response_manifest,
            sjt_records=sjt_records,
            neo_records=neo_records,
            persona_modes=persona_modes,
        )
    active_persona_mode = persona_modes[0]
    sjt_records = _filter_persona_mode_records(
        sjt_records,
        persona_mode=active_persona_mode,
    )
    neo_records = _filter_persona_mode_records(
        neo_records,
        persona_mode=active_persona_mode,
    )
    sjt_long, sjt_wide = _score_sjt(
        sjt_records,
        item_order=item_order,
        items=items,
        expected_respondent_count=expected_respondents,
    )
    neo_long, neo_dimensions = _score_neo(
        neo_records,
        expected_respondents=list(sjt_wide.index),
    )
    item_statistics, item_frame, option_frame = _item_statistics(
        sjt_long,
        sjt_wide,
        item_order=item_order,
        items=items,
    )
    scale_statistics = _scale_statistics(sjt_wide, items)
    respondent_scores = _respondent_scores(
        sjt_wide,
        neo_dimensions,
        items,
    )
    validity_statistics = _validity_statistics(
        respondent_scores,
        criterion=criterion,
    )
    item_statistics, item_quality_frame = _enrich_item_quality(
        item_statistics,
        item_order=item_order,
    )
    measurement_evaluation = _evaluate_test_quality(
        item_statistics=item_statistics,
        scale_statistics=scale_statistics,
        validity_statistics=validity_statistics,
        criterion=criterion,
    )

    output_dir = manifest_path.parent / "psychometrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_sjt_path = output_dir / "scored_sjt_responses.csv"
    scored_neo_path = output_dir / "scored_neo_ffi_responses.csv"
    respondent_scores_path = output_dir / "respondent_scores.csv"
    item_statistics_path = output_dir / "item_statistics.csv"
    item_quality_path = output_dir / "item_quality.csv"
    option_statistics_path = output_dir / "option_statistics.csv"
    scale_statistics_path = output_dir / "scale_statistics.json"
    validity_statistics_path = output_dir / "validity_statistics.json"
    measurement_evaluation_path = (
        output_dir / "measurement_evaluation.json"
    )
    analysis_manifest_path = output_dir / "analysis_manifest.json"

    _write_csv_atomic(scored_sjt_path, sjt_long)
    _write_csv_atomic(scored_neo_path, neo_long)
    _write_csv_atomic(respondent_scores_path, respondent_scores)
    _write_csv_atomic(item_statistics_path, item_frame)
    _write_csv_atomic(item_quality_path, item_quality_frame)
    _write_csv_atomic(option_statistics_path, option_frame)
    _write_json_atomic(scale_statistics_path, scale_statistics)
    _write_json_atomic(validity_statistics_path, validity_statistics)
    _write_json_atomic(
        measurement_evaluation_path,
        measurement_evaluation,
    )

    qualified_count = sum(
        bool(item["qualification"]["qualified"])
        for item in item_statistics.values()
    )
    warnings = []
    if expected_respondents < 100:
        warnings.append(
            "样本量少于100，项目统计、信度和相关系数可能有较大抽样波动。"
        )
    if expected_respondents == 30:
        warnings.append(
            "30人样本中1次选择率为3.33%、2次为6.67%，"
            "5%选项阈值的判定较粗糙。"
        )
    output_files = {
        "scored_sjt_responses": str(scored_sjt_path.resolve()),
        "scored_neo_ffi_responses": str(scored_neo_path.resolve()),
        "respondent_scores": str(respondent_scores_path.resolve()),
        "item_statistics": str(item_statistics_path.resolve()),
        "item_quality": str(item_quality_path.resolve()),
        "option_statistics": str(option_statistics_path.resolve()),
        "scale_statistics": str(scale_statistics_path.resolve()),
        "validity_statistics": str(validity_statistics_path.resolve()),
        "measurement_evaluation": str(
            measurement_evaluation_path.resolve()
        ),
        "analysis_manifest": str(analysis_manifest_path.resolve()),
    }
    analysis_manifest = {
        "schema_version": 1,
        "status": "completed",
        "formula_version": PSYCHOMETRIC_FORMULA_VERSION,
        "evaluation_version": MEASUREMENT_EVALUATION_VERSION,
        "run_id": state["run_id"],
        "item_bank_id": state["item_bank_id"],
        "item_bank_version": state["item_bank_version"],
        "item_bank_fingerprint": state["item_bank_fingerprint"],
        "sample_size": expected_respondents,
        "persona_mode": active_persona_mode,
        "criterion": criterion,
        "item_count": len(item_order),
        "qualified_item_count": qualified_count,
        "qualification_rate": qualified_count / len(item_order),
        "criteria": {
            "difficulty_range": [
                DIFFICULTY_LOWER_BOUND,
                DIFFICULTY_UPPER_BOUND,
            ],
            "citc_minimum_acceptable": CITC_MINIMUM_ACCEPTABLE,
            "citc_revision_threshold": CITC_REVISION_THRESHOLD,
            "minimum_option_rate": MINIMUM_OPTION_RATE,
            "quality_evaluation": {
                "citc_minimum_acceptable": CITC_MINIMUM_ACCEPTABLE,
                "citc_revision_threshold": CITC_REVISION_THRESHOLD,
                "citc_strong": CITC_STRONG,
                "statistical_warning_triggers_revision": False,
                "statistical_warning_triggers_removal": False,
            },
        },
        "formulas": {
            "difficulty": "(item_mean - theoretical_min) / (theoretical_max - theoretical_min)",
            "facet_citc": (
                "Pearson correlation(item, sum of the other items with the "
                "same target_dimension_id)"
            ),
            "multiple_comparison_adjustment": "Benjamini-Hochberg FDR",
            "cronbach_alpha": "k/(k-1) * (1 - sum(item_sample_variances)/total_sample_variance)",
            "neo_reverse_scoring": "6 - raw_response",
            "convergent_validity": "Spearman rank correlation",
        },
        "missing_value_policy": "complete response matrices required; no imputation",
        "input_files": {
            "response_manifest": {
                "path": str(manifest_path),
                "sha256": _file_sha256(manifest_path),
            },
            "sjt_responses": {
                "path": str(sjt_path),
                "sha256": _file_sha256(sjt_path),
            },
            "neo_ffi_responses": {
                "path": str(neo_path),
                "sha256": _file_sha256(neo_path),
            },
        },
        "output_files": output_files,
        "warnings": warnings,
        "completed_at": utc_timestamp(),
    }
    _write_json_atomic(analysis_manifest_path, analysis_manifest)

    test_statistics = {
        **scale_statistics,
        "persona_mode": active_persona_mode,
        "criterion": criterion,
        "qualified_item_count": qualified_count,
        "qualification_rate": qualified_count / len(item_order),
        "qualification_criteria": analysis_manifest["criteria"],
        "convergent_validity": validity_statistics,
        "measurement_evaluation": measurement_evaluation,
        "warnings": warnings,
        "output_files": output_files,
        "formula_version": PSYCHOMETRIC_FORMULA_VERSION,
        "evaluation_version": MEASUREMENT_EVALUATION_VERSION,
    }
    recommendation_counts = measurement_evaluation[
        "item_recommendation_counts"
    ]
    return {
        "state_update": {
            "item_statistics": item_statistics,
            "test_statistics": _json_safe(test_statistics),
        },
        "summary": (
            f"已完成 {expected_respondents} 名被试、{len(item_order)} 道题的"
            f"经典测量分析；评价建议为保留"
            f"{recommendation_counts.get('retain', 0)}题、修改"
            f"{recommendation_counts.get('revise', 0)}题、删除"
            f"{recommendation_counts.get('remove', 0)}题；"
            f"Cronbach α={scale_statistics['cronbach_alpha']}。"
        ),
    }


def run_saved_psychometric_analysis(
    manifest_path: str | Path,
    *,
    scoring_snapshot_path: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze a completed saved run without requiring an in-memory checkpoint."""

    resolved_manifest = Path(manifest_path).resolve()
    manifest = _read_json(resolved_manifest)
    resolved_snapshot = Path(
        scoring_snapshot_path
        or manifest.get("scoring_snapshot_path")
        or resolved_manifest.parent / "scoring_snapshot.json"
    ).resolve()
    if not resolved_snapshot.is_file():
        raise ValueError(
            "保存的虚拟作答缺少 scoring_snapshot.json，"
            "无法在脱离工作流 State 后恢复计分键"
        )
    snapshot = _read_json(resolved_snapshot)
    for field in (
        "run_id",
        "item_bank_id",
        "item_bank_version",
        "item_bank_fingerprint",
    ):
        if snapshot.get(field) != manifest.get(field):
            raise ValueError(
                f"scoring snapshot 的 {field} 与作答 manifest 不一致"
            )
    items = snapshot.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("scoring snapshot 没有冻结题目")

    state = create_initial_state("分析已保存的虚拟作答")
    state.update(
        {
            "run_id": manifest["run_id"],
            "item_bank_id": manifest["item_bank_id"],
            "item_bank_version": manifest["item_bank_version"],
            "item_bank_fingerprint": manifest["item_bank_fingerprint"],
            "frozen_item_bank": items,
            "virtual_response_data_ref": str(resolved_manifest),
            "virtual_response_item_bank_id": manifest["item_bank_id"],
            "virtual_response_item_bank_version": manifest[
                "item_bank_version"
            ],
        }
    )
    return run_psychometric_analysis(state)
