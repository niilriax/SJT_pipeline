"""Deterministic scoring and psychometric analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from sjt_system.evaluation.respondents import PERSONA_MODE_SUMMARY_PLUS_ITEMS
from sjt_system.state import PSJTState, create_initial_state
from sjt_system.runtime.trace import utc_timestamp


PSYCHOMETRIC_FORMULA_VERSION = "sjt-ctt-v5"
MEASUREMENT_EVALUATION_VERSION = "sjt-evaluation-v5"
DIFFICULTY_LOWER_BOUND = 0.20
DIFFICULTY_UPPER_BOUND = 0.80
MINIMUM_OPTION_RATE = 0.05
CITC_MINIMUM_ACCEPTABLE = 0.30
CITC_REVISION_THRESHOLD = 0.20
CITC_STRONG = 0.50


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
    if not sjt_path.is_file() or not neo_path.is_file():
        raise ValueError("虚拟作答目录缺少 SJT 或 Neo-FFI JSONL 文件")
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
    criterion = _resolve_analysis_criterion(state, response_manifest)

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
