"""Concurrent virtual-response generation and persistence."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from sjt_system.agent.client import (
    get_model,
    get_model_request_timeout_seconds,
    with_compatible_structured_output,
)
from sjt_system.authoring.bank import build_virtual_response_context
from sjt_system.authoring.construct_registry import (
    resolve_neo_ffi_criterion,
)
from sjt_system.runtime.progress import emit_progress
from sjt_system.runtime.io import write_json_atomic as _write_json_atomic
from sjt_system.evaluation.respondents import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    MAX_ALLOWED_CONCURRENCY,
    PERSONA_MODE_SUMMARY_PLUS_ITEMS,
    SUPPORTED_PERSONA_MODES,
    resolve_virtual_respondent_profiles,
)
from sjt_system.state import PSJTState
from sjt_system.runtime.trace import utc_timestamp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEO_FFI_PATH = PROJECT_ROOT / "docs" / "Neo-FFI.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "virtual_responses"
VIRTUAL_RESPONSE_PROMPT_VERSION = "liu-summary-persona-sjt-neo-v3"
PERSONA_SUMMARY_PROMPT_VERSION = "liu-item-response-summary-v2"


class SJTSelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_option_id: str


class NeoFFIBatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ratings: list[int]


class PersonaSummaryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str


def load_neo_ffi(
    path: str | Path = DEFAULT_NEO_FFI_PATH,
) -> list[dict[str, Any]]:
    """读取五个维度、每个维度12题的 Neo-FFI 目标量表。"""

    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or list(raw) != ["N", "E", "O", "A", "C"]:
        raise ValueError("Neo-FFI 必须按 N、E、O、A、C 五个维度组织")

    dimensions = []
    for dimension_code, dimension_data in raw.items():
        if not isinstance(dimension_data, dict):
            raise ValueError(f"Neo-FFI 维度 {dimension_code} 结构无效")
        raw_items = dimension_data.get("items")
        if not isinstance(raw_items, dict) or len(raw_items) != 12:
            raise ValueError(
                f"Neo-FFI 维度 {dimension_code} 必须包含12道题"
            )
        items = []
        for item_id, item_data in raw_items.items():
            if not isinstance(item_data, dict):
                raise ValueError(f"Neo-FFI 题目 {item_id} 结构无效")
            text = item_data.get("item")
            scoring = item_data.get("scoring")
            if not isinstance(text, str) or not text:
                raise ValueError(f"Neo-FFI 题目 {item_id} 缺少题干")
            if scoring not in {"+", "-"}:
                raise ValueError(f"Neo-FFI 题目 {item_id} 计分方向无效")
            items.append(
                {
                    "item_id": item_id,
                    "text": text,
                    "scoring_direction": scoring,
                }
            )
        dimensions.append(
            {
                "dimension_code": dimension_code,
                "domain": dimension_data.get("domain"),
                "items": items,
            }
        )
    return dimensions


def _personality_response_lines(profile: Mapping[str, Any]) -> list[str]:
    personality_items = profile.get("personality_items")
    if not isinstance(personality_items, list) or not personality_items:
        raise ValueError("虚拟被试缺少人格逐题作答")
    lines = []
    for item in personality_items:
        if not isinstance(item, Mapping):
            raise ValueError("虚拟被试包含无效人格题目")
        statement = item.get("statement")
        response_label = item.get("response_label")
        if not isinstance(statement, str) or not isinstance(
            response_label,
            str,
        ):
            raise ValueError("虚拟被试人格题目缺少题干或作答标签")
        lines.append(f"{statement}：{response_label}")
    return lines


def build_persona_summary_messages(
    profile: Mapping[str, Any],
) -> list[tuple[str, str]]:
    """Build a summary using only the respondent's item-level answers."""

    lines = _personality_response_lines(profile)
    system_message = (
        "You summarize recurring behavioral response patterns from a virtual "
        "respondent's item-level questionnaire responses. Use only the supplied "
        "statements and response labels. Distinguish directly observed response "
        "patterns from broader inferences, and do not infer beyond what the "
        "responses support. Describe when the person tends to respond in a certain "
        "way, meaningful tensions across situations, and uncertainty. Preserve "
        "contradictions instead of forcing a perfectly coherent profile. Do not "
        "name or restate latent traits, facets, target constructs, or global "
        "personality labels; do not make evaluative judgments. Do not invent "
        "demographics, biography, motives, abilities, resources, diagnoses, life "
        "events, or causal explanations. Do not mention questionnaire names, item "
        "codes, numeric scores, scoring direction, or psychometric jargon. Use "
        "calibrated language such as 'tends to', 'may', or 'in some situations'. "
        "Write one compact Simplified-Chinese paragraph of 100-180 Chinese "
        "characters. Return JSON only: {\"summary\":\"...\"}."
    )
    user_message = (
        "Item-level responses (the only evidence available):\n"
        + "\n".join(lines)
    )
    return [("system", system_message), ("human", user_message)]


def build_persona_prompt(
    profile: Mapping[str, Any],
    *,
    persona_mode: str = PERSONA_MODE_SUMMARY_PLUS_ITEMS,
    personality_summary: str | None = None,
) -> str:
    """Build the summary-plus-item persona instructions."""

    if persona_mode not in SUPPORTED_PERSONA_MODES:
        raise ValueError(f"不支持的人格提示模式：{persona_mode}")
    lines = _personality_response_lines(profile)
    if persona_mode == PERSONA_MODE_SUMMARY_PLUS_ITEMS:
        if not isinstance(personality_summary, str) or not (
            personality_summary.strip()
        ):
            raise ValueError("summary_plus_items 模式缺少人格总结")
        summary_section = (
            "\n\n补充人格总结：\n"
            f"{personality_summary.strip()}\n"
            "该总结只用于帮助整合上述逐题作答。若总结与逐题作答发生冲突，"
            "应以逐题作答为准；不要据此补充任何未提供的个人背景。"
        )
    else:
        summary_section = ""
    return (
        "请想象你正在扮演一个特定的人。\n"
        "下面是这个人的人格逐题作答。每项作答使用五级："
        "非常不同意、比较不同意、不确定、比较同意、非常同意。\n"
        "这些逐题作答是判断此人人格的主要且权威的依据。\n\n"
        + "\n".join(lines)
        + summary_section
    )


def build_sjt_messages(
    persona_prompt: str,
    item: Mapping[str, Any],
) -> list[tuple[str, str]]:
    """构造一次只回答一道 SJT 题、且不含评分信息的消息。"""

    options = item.get("response_options")
    if not isinstance(options, list) or not options:
        raise ValueError("SJT 题目缺少作答选项")
    option_lines = []
    for option in options:
        if not isinstance(option, Mapping):
            raise ValueError("SJT 题目包含无效选项")
        option_id = option.get("option_id")
        text = option.get("text")
        if not isinstance(option_id, str) or not isinstance(text, str):
            raise ValueError("SJT 选项缺少 option_id 或文本")
        option_lines.append(f"{option_id}. {text}")

    system_message = (
        persona_prompt
        + "\n\n现在请以这个人的真实行为倾向回答一道情境判断题。"
        "请选择此人最可能采取的行为，而不是理论上最好、最正确或"
        "社会赞许程度最高的行为。每次作答相互独立。"
        "只返回一个JSON对象，不要解释，格式为："
        '{"selected_option_id":"选项编号"}。'
    )
    user_message = (
        f"情境：\n{item.get('scenario', '')}\n\n"
        f"作答要求：\n{item.get('response_instruction', '')}\n\n"
        "选项：\n"
        + "\n".join(option_lines)
    )
    return [("system", system_message), ("human", user_message)]


def build_neo_ffi_messages(
    persona_prompt: str,
    items: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    """构造一个不暴露维度名称和正反向计分的12题批次。"""

    description_lines = [
        f"描述 {index}：{item['text']}"
        for index, item in enumerate(items, 1)
    ]
    system_message = (
        persona_prompt
        + "\n\n请判断你扮演的这个人对下面每项描述的同意程度："
        "1=非常不同意，2=比较不同意，3=不确定，4=比较同意，"
        "5=非常同意。按题目顺序返回12个整数。"
        "只返回一个JSON对象，不要解释，格式为："
        '{"ratings":[1,2,3,4,5,1,2,3,4,5,1,2]}。'
    )
    return [
        ("system", system_message),
        ("human", "\n".join(description_lines)),
    ]


def _structured_result_dict(result: object) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    legacy_dict = getattr(result, "dict", None)
    if callable(legacy_dict):
        dumped = legacy_dict()
        if isinstance(dumped, dict):
            return dumped
    raise ValueError("模型没有返回有效的结构化对象")


async def _invoke_with_retry(
    runnable: Any,
    messages: list[tuple[str, str]],
    *,
    semaphore: asyncio.Semaphore,
    validator: Any,
    max_retries: int,
    retry_delay_seconds: float,
    request_timeout_seconds: float,
    job_label: str,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                raw_result = await asyncio.wait_for(
                    runnable.ainvoke(messages),
                    timeout=request_timeout_seconds,
                )
            return validator(_structured_result_dict(raw_result))
        except Exception as exc:
            if isinstance(exc, TimeoutError):
                last_error = TimeoutError(
                    f"单次请求超过 {request_timeout_seconds:g} 秒"
                )
            else:
                last_error = exc
            if attempt >= max_retries:
                break
            emit_progress(
                {
                    "type": "request_retry",
                    "retry_kind": (
                        "network_timeout"
                        if isinstance(exc, TimeoutError)
                        else "output_repair"
                    ),
                    "job_label": job_label,
                    "attempt": attempt + 2,
                    "max_attempts": max_retries + 1,
                    "reason": str(last_error),
                }
            )
            await asyncio.sleep(retry_delay_seconds * (2**attempt))
    raise RuntimeError(
        f"{job_label} 在 {max_retries + 1} 次尝试后仍失败：{last_error}"
    ) from last_error


def _append_jsonl_records(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            )
        )


def _load_jsonl_keys(
    path: Path,
    key_fields: Sequence[str],
) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    keys = set()
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
                    f"{path.name} 第 {line_number} 行不是对象"
                )
            keys.add(tuple(record.get(field) for field in key_fields))
    return keys


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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
                    f"{path.name} 第 {line_number} 行不是对象"
                )
            records.append(record)
    return records


def _resolve_persona_modes(config: Mapping[str, Any]) -> list[str]:
    """Resolve configured modes while keeping old checkpoints readable."""

    raw_modes = config.get("persona_modes")
    if raw_modes is None:
        return [PERSONA_MODE_SUMMARY_PLUS_ITEMS]
    if (
        not isinstance(raw_modes, Sequence)
        or isinstance(raw_modes, (str, bytes))
        or not raw_modes
    ):
        raise ValueError("virtual_sample_config.persona_modes 必须是非空列表")
    modes = list(raw_modes)
    if (
        len(set(modes)) != len(modes)
        or any(mode not in SUPPORTED_PERSONA_MODES for mode in modes)
    ):
        raise ValueError(
            "persona_modes 只能包含不重复的 summary_plus_items"
        )
    return modes


def _seed_incremental_response_files(
    *,
    state: PSJTState,
    respondent_refs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    sjt_items: Sequence[Mapping[str, Any]],
    model_id: str,
    criterion: Mapping[str, str],
    persona_modes: Sequence[str],
    summary_path: Path,
    sjt_path: Path,
    neo_path: Path,
) -> dict[str, Any]:
    """Reuse unchanged-item and Neo-FFI records from the previous bank."""

    source_ref = state.get("previous_virtual_response_data_ref")
    if not isinstance(source_ref, str) or not source_ref:
        return {
            "source_manifest_path": None,
            "reused_persona_summary_records": 0,
            "reused_sjt_records": 0,
            "reused_neo_ffi_records": 0,
        }
    source_manifest_path = Path(source_ref).resolve()
    if not source_manifest_path.is_file():
        return {
            "source_manifest_path": None,
            "reused_persona_summary_records": 0,
            "reused_sjt_records": 0,
            "reused_neo_ffi_records": 0,
        }
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    compatible = (
        source_manifest.get("status") == "completed"
        and source_manifest.get("run_id") == state.get("run_id")
        and source_manifest.get("pool_id") == config.get("pool_id")
        and source_manifest.get("source_sha256")
        == config.get("source_sha256")
        and source_manifest.get("sample_size") == len(respondent_refs)
        and source_manifest.get("prompt_version")
        == VIRTUAL_RESPONSE_PROMPT_VERSION
        and source_manifest.get("model_id") == model_id
        and source_manifest.get("persona_modes") == list(persona_modes)
        and source_manifest.get("persona_summary_prompt_version")
        == PERSONA_SUMMARY_PROMPT_VERSION
        and source_manifest.get("criterion_domain_id")
        == criterion["domain_id"]
        and source_manifest.get("criterion_neo_ffi_dimension")
        == criterion["neo_ffi_dimension"]
    )
    if not compatible:
        return {
            "source_manifest_path": None,
            "reused_persona_summary_records": 0,
            "reused_sjt_records": 0,
            "reused_neo_ffi_records": 0,
        }

    respondent_ids = {
        str(reference.get("respondent_id"))
        for reference in respondent_refs
        if isinstance(reference.get("respondent_id"), str)
    }
    current_versions = {
        str(item.get("item_id")): item.get("item_version")
        for item in sjt_items
        if isinstance(item.get("item_id"), str)
    }
    source_sjt_path = source_manifest_path.parent / "sjt_responses.jsonl"
    source_neo_path = (
        source_manifest_path.parent / "neo_ffi_responses.jsonl"
    )
    source_summary_path = (
        source_manifest_path.parent / "persona_summaries.jsonl"
    )
    source_summaries = _load_jsonl_records(source_summary_path)
    source_sjt = _load_jsonl_records(source_sjt_path)
    source_neo = _load_jsonl_records(source_neo_path)

    reused_summaries: list[dict[str, Any]] = []
    seen_summaries: set[str] = set()
    if persona_modes == [PERSONA_MODE_SUMMARY_PLUS_ITEMS]:
        for source_record in source_summaries:
            respondent_id = source_record.get("respondent_id")
            if (
                respondent_id not in respondent_ids
                or respondent_id in seen_summaries
            ):
                continue
            if (
                source_record.get("prompt_version")
                != PERSONA_SUMMARY_PROMPT_VERSION
                or source_record.get("model_id") != model_id
            ):
                continue
            summary = source_record.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                continue
            seen_summaries.add(respondent_id)
            reused_summaries.append(
                {
                    **source_record,
                    "response_source": "reused_persona_summary",
                    "source_item_bank_id": source_manifest.get(
                        "item_bank_id"
                    ),
                    "source_item_bank_version": source_manifest.get(
                        "item_bank_version"
                    ),
                }
            )

    reused_sjt: list[dict[str, Any]] = []
    seen_sjt: set[tuple[str, str, str, Any]] = set()
    for source_record in source_sjt:
        respondent_id = source_record.get("respondent_id")
        item_id = source_record.get("item_id")
        version = source_record.get("item_version")
        persona_mode = source_record.get("persona_mode")
        key = (
            str(respondent_id),
            str(persona_mode),
            str(item_id),
            version,
        )
        if (
            respondent_id not in respondent_ids
            or persona_mode not in persona_modes
            or item_id not in current_versions
            or version != current_versions[item_id]
            or key in seen_sjt
        ):
            continue
        seen_sjt.add(key)
        reused_sjt.append(
            {
                **source_record,
                "item_bank_id": state["item_bank_id"],
                "item_bank_version": state["item_bank_version"],
                "item_bank_fingerprint": state.get(
                    "item_bank_fingerprint"
                ),
                "response_source": "reused_unchanged_item",
                "source_item_bank_id": source_manifest.get(
                    "item_bank_id"
                ),
                "source_item_bank_version": source_manifest.get(
                    "item_bank_version"
                ),
            }
        )

    reused_neo: list[dict[str, Any]] = []
    seen_neo: set[tuple[str, str, str]] = set()
    for source_record in source_neo:
        respondent_id = source_record.get("respondent_id")
        item_id = source_record.get("item_id")
        persona_mode = source_record.get("persona_mode")
        key = (str(respondent_id), str(persona_mode), str(item_id))
        if (
            respondent_id not in respondent_ids
            or persona_mode not in persona_modes
            or key in seen_neo
        ):
            continue
        seen_neo.add(key)
        reused_neo.append(
            {
                **source_record,
                "item_bank_id": state["item_bank_id"],
                "item_bank_version": state["item_bank_version"],
                "item_bank_fingerprint": state.get(
                    "item_bank_fingerprint"
                ),
                "response_source": "reused_neo_ffi",
                "source_item_bank_id": source_manifest.get(
                    "item_bank_id"
                ),
                "source_item_bank_version": source_manifest.get(
                    "item_bank_version"
                ),
            }
        )

    if reused_summaries:
        _append_jsonl_records(summary_path, reused_summaries)
    if reused_sjt:
        _append_jsonl_records(sjt_path, reused_sjt)
    if reused_neo:
        _append_jsonl_records(neo_path, reused_neo)
    return {
        "source_manifest_path": str(source_manifest_path),
        "reused_persona_summary_records": len(reused_summaries),
        "reused_sjt_records": len(reused_sjt),
        "reused_neo_ffi_records": len(reused_neo),
    }


def _simulation_signature(
    *,
    state: PSJTState,
    respondent_refs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    persona_modes: Sequence[str],
    criterion: Mapping[str, str],
    model_id: str,
) -> str:
    payload = {
        "run_id": state["run_id"],
        "item_bank_id": state["item_bank_id"],
        "item_bank_version": state["item_bank_version"],
        "item_bank_fingerprint": state.get("item_bank_fingerprint"),
        "respondents": list(respondent_refs),
        "pool_id": config.get("pool_id"),
        "source_sha256": config.get("source_sha256"),
        "persona_modes": list(persona_modes),
        "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
        "persona_summary_prompt_version": PERSONA_SUMMARY_PROMPT_VERSION,
        "criterion": dict(criterion),
        "model_id": model_id,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


class VirtualResponseRunner:
    """以受控并发执行独立 SJT 单题和 Neo-FFI 维度批次。"""

    def __init__(
        self,
        *,
        base_model: Any | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay_seconds: float = 1.0,
        request_timeout_seconds: float | None = None,
    ) -> None:
        if (
            not isinstance(max_concurrency, int)
            or isinstance(max_concurrency, bool)
            or max_concurrency < 1
            or max_concurrency > MAX_ALLOWED_CONCURRENCY
        ):
            raise ValueError(
                "max_concurrency 必须是 1 到 "
                f"{MAX_ALLOWED_CONCURRENCY} 之间的整数"
            )
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or max_retries < 0
            or max_retries > 5
        ):
            raise ValueError("max_retries 必须是 0 到 5 之间的整数")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds 不能为负数")
        if request_timeout_seconds is None:
            request_timeout_seconds = get_model_request_timeout_seconds()
        if (
            not isinstance(request_timeout_seconds, (int, float))
            or isinstance(request_timeout_seconds, bool)
            or request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds 必须是正数")

        self.base_model = base_model or get_model()
        self.sjt_model, self.structured_output_method = (
            with_compatible_structured_output(
                self.base_model,
                SJTSelectionOutput,
            )
        )
        self.neo_ffi_model, neo_method = (
            with_compatible_structured_output(
                self.base_model,
                NeoFFIBatchOutput,
            )
        )
        self.persona_summary_model, summary_method = (
            with_compatible_structured_output(
                self.base_model,
                PersonaSummaryOutput,
            )
        )
        if (
            neo_method != self.structured_output_method
            or summary_method != self.structured_output_method
        ):
            raise ValueError("同一模型的结构化输出方式不一致")
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.write_lock = asyncio.Lock()
        self.model_id = (
            getattr(self.base_model, "model_name", None)
            or getattr(self.base_model, "model", None)
            or os.getenv("MODEL_ID")
            or "unknown"
        )

    async def _append_record(
        self,
        path: Path,
        record: Mapping[str, Any],
    ) -> None:
        async with self.write_lock:
            _append_jsonl_records(path, [record])

    async def _append_records(
        self,
        path: Path,
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        async with self.write_lock:
            _append_jsonl_records(path, records)

    async def run(
        self,
        *,
        state: PSJTState,
        output_dir: Path,
        neo_ffi_path: str | Path = DEFAULT_NEO_FFI_PATH,
    ) -> dict[str, Any]:
        profile = state.get("construct_profile")
        if not isinstance(profile, Mapping):
            blueprint = state.get("blueprint")
            if isinstance(blueprint, Mapping):
                candidate = blueprint.get("construct_profile_snapshot")
                if isinstance(candidate, Mapping):
                    profile = candidate
        criterion = resolve_neo_ffi_criterion(
            construct_profile=profile,
        )
        context = build_virtual_response_context(state)
        config = context.get("virtual_sample_config")
        respondent_refs = context.get("virtual_respondents")
        if not isinstance(config, Mapping):
            raise ValueError("虚拟作答前必须先配置虚拟样本")
        if not isinstance(respondent_refs, list) or not respondent_refs:
            raise ValueError("虚拟作答前必须先选择虚拟被试")
        if config.get("sample_size") != len(respondent_refs):
            raise ValueError("虚拟样本配置人数与被试引用数量不一致")

        profiles = resolve_virtual_respondent_profiles(respondent_refs)
        profile_by_id = {
            profile["respondent_id"]: profile for profile in profiles
        }
        persona_modes = _resolve_persona_modes(config)
        summaries_required = (
            PERSONA_MODE_SUMMARY_PLUS_ITEMS in persona_modes
        )
        sjt_items = context["items"]
        neo_dimensions = load_neo_ffi(neo_ffi_path)

        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "persona_summaries.jsonl"
        sjt_path = output_dir / "sjt_responses.jsonl"
        neo_path = output_dir / "neo_ffi_responses.jsonl"
        manifest_path = output_dir / "manifest.json"
        scoring_snapshot_path = output_dir / "scoring_snapshot.json"
        signature = _simulation_signature(
            state=state,
            respondent_refs=respondent_refs,
            config=config,
            persona_modes=persona_modes,
            criterion=criterion,
            model_id=self.model_id,
        )
        if manifest_path.exists():
            existing_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if existing_manifest.get("simulation_signature") != signature:
                raise ValueError(
                    "输出目录已包含不同配置的虚拟作答，拒绝混合数据"
                )
        elif summary_path.exists() or sjt_path.exists() or neo_path.exists():
            raise ValueError("输出目录存在作答文件但缺少 manifest.json")

        reuse_summary = {
            "source_manifest_path": None,
            "reused_persona_summary_records": 0,
            "reused_sjt_records": 0,
            "reused_neo_ffi_records": 0,
        }
        if not manifest_path.exists():
            reuse_summary = _seed_incremental_response_files(
                state=state,
                respondent_refs=respondent_refs,
                config=config,
                sjt_items=sjt_items,
                model_id=self.model_id,
                criterion=criterion,
                persona_modes=persona_modes,
                summary_path=summary_path,
                sjt_path=sjt_path,
                neo_path=neo_path,
            )
        summary_keys = _load_jsonl_keys(
            summary_path,
            ("respondent_id",),
        )
        sjt_keys = _load_jsonl_keys(
            sjt_path,
            ("respondent_id", "persona_mode", "item_id", "item_version"),
        )
        neo_keys = _load_jsonl_keys(
            neo_path,
            ("respondent_id", "persona_mode", "item_id"),
        )
        initial_summary_count = len(summary_keys)
        initial_sjt_count = len(sjt_keys)
        initial_neo_count = len(neo_keys)
        resumed_summary_count = (
            initial_summary_count
            - reuse_summary["reused_persona_summary_records"]
        )
        resumed_sjt_count = (
            initial_sjt_count - reuse_summary["reused_sjt_records"]
        )
        resumed_neo_count = (
            initial_neo_count
            - reuse_summary["reused_neo_ffi_records"]
        )
        expected_summary_records = (
            len(respondent_refs) if summaries_required else 0
        )
        expected_sjt_records = (
            len(respondent_refs) * len(sjt_items) * len(persona_modes)
        )
        expected_neo_records = (
            len(respondent_refs) * 60 * len(persona_modes)
        )
        scheduled_summary_calls = (
            expected_summary_records - initial_summary_count
        )
        scheduled_sjt_calls = expected_sjt_records - initial_sjt_count
        scheduled_neo_calls = (
            expected_neo_records - initial_neo_count
        ) // 12
        progress_lock = asyncio.Lock()
        completed_summary_calls = 0
        completed_sjt_calls = 0
        completed_neo_calls = 0

        def should_report_progress(completed: int, total: int) -> bool:
            if total <= 0:
                return completed == 0
            interval = max(1, total // 20)
            return completed in {1, total} or completed % interval == 0

        async def report_call_completed(kind: str) -> None:
            nonlocal completed_summary_calls
            nonlocal completed_sjt_calls, completed_neo_calls
            async with progress_lock:
                if kind == "summary":
                    completed_summary_calls += 1
                    completed = completed_summary_calls
                    total = scheduled_summary_calls
                    label = "PersonaSummary"
                elif kind == "sjt":
                    completed_sjt_calls += 1
                    completed = completed_sjt_calls
                    total = scheduled_sjt_calls
                    label = "SJT"
                else:
                    completed_neo_calls += 1
                    completed = completed_neo_calls
                    total = scheduled_neo_calls
                    label = "Neo-FFI"
                if should_report_progress(completed, total):
                    emit_progress(
                        {
                            "type": "simulation_progress",
                            "stage": label,
                            "completed": completed,
                            "total": total,
                        }
                    )

        manifest = {
            "schema_version": 1,
            "status": "in_progress",
            "simulation_signature": signature,
            "run_id": state["run_id"],
            "item_bank_id": state["item_bank_id"],
            "item_bank_version": state["item_bank_version"],
            "item_bank_fingerprint": state.get("item_bank_fingerprint"),
            "pool_id": config.get("pool_id"),
            "source_sha256": config.get("source_sha256"),
            "criterion_domain_id": criterion["domain_id"],
            "criterion_domain_name": criterion["domain_name_en"],
            "criterion_neo_ffi_dimension": criterion[
                "neo_ffi_dimension"
            ],
            "sample_size": len(respondent_refs),
            "persona_modes": persona_modes,
            "persona_mode_count": len(persona_modes),
            "sjt_item_count": len(sjt_items),
            "neo_ffi_item_count": 60,
            "sjt_response_mode": "single_item_independent_call",
            "neo_ffi_response_mode": "five_dimension_batches_of_12",
            "model_id": self.model_id,
            "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
            "persona_summary_prompt_version": (
                PERSONA_SUMMARY_PROMPT_VERSION
            ),
            "max_concurrency": self.max_concurrency,
            "max_retries": self.max_retries,
            "request_timeout_seconds": self.request_timeout_seconds,
            "expected_persona_summary_records": expected_summary_records,
            "expected_sjt_records": expected_sjt_records,
            "expected_neo_ffi_records": expected_neo_records,
            **reuse_summary,
            "scoring_snapshot_path": str(scoring_snapshot_path.resolve()),
            "persona_summary_path": str(summary_path.resolve()),
            "interpretation_limitations": [
                (
                    "Neo-FFI responses are model predictions from the same "
                    "persona inputs, not independent human criterion data."
                ),
                (
                    "Semantic overlap between persona items and Neo-FFI items "
                    "may inflate apparent consistency."
                ),
                (
                    "The generated summary is derived from the same item-level "
                    "responses and can amplify apparent profile coherence; "
                    "summary_plus_items is the only active persona mode."
                ),
            ],
            "started_at": utc_timestamp(),
        }
        _write_json_atomic(
            scoring_snapshot_path,
            {
                "schema_version": 1,
                "run_id": state["run_id"],
                "item_bank_id": state["item_bank_id"],
                "item_bank_version": state["item_bank_version"],
                "item_bank_fingerprint": state.get(
                    "item_bank_fingerprint"
                ),
                "criterion_domain_id": criterion["domain_id"],
                "criterion_domain_name": criterion["domain_name_en"],
                "criterion_neo_ffi_dimension": criterion[
                    "neo_ffi_dimension"
                ],
                "items": [
                    {
                        "item_id": item.get("item_id"),
                        "version": item.get("version"),
                        "target_dimension_id": item.get(
                            "target_dimension_id"
                        ),
                        "context_category": item.get("context_category"),
                        "option_ids": [
                            option.get("option_id")
                            for option in item.get("response_options") or []
                            if isinstance(option, Mapping)
                        ],
                        "scoring_key": dict(item.get("scoring_key") or {}),
                    }
                    for item in state["frozen_item_bank"]
                ],
            },
        )
        _write_json_atomic(manifest_path, manifest)

        def validate_persona_summary(
            result: Mapping[str, Any],
        ) -> str:
            summary = result.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("人格总结必须是非空字符串")
            normalized = summary.strip()
            if len(normalized) > 500:
                raise ValueError("人格总结超过 500 个字符")
            return normalized

        async def run_summary_job(
            respondent_ref: Mapping[str, Any],
        ) -> None:
            respondent_id = respondent_ref["respondent_id"]
            key = (respondent_id,)
            if key in summary_keys:
                return
            summary = await _invoke_with_retry(
                self.persona_summary_model,
                build_persona_summary_messages(
                    profile_by_id[respondent_id]
                ),
                semaphore=self.semaphore,
                validator=validate_persona_summary,
                max_retries=self.max_retries,
                retry_delay_seconds=self.retry_delay_seconds,
                request_timeout_seconds=self.request_timeout_seconds,
                job_label=f"PersonaSummary {respondent_id}",
            )
            record = {
                "record_type": "persona_summary",
                "run_id": state["run_id"],
                "respondent_id": respondent_id,
                "pool_id": config.get("pool_id"),
                "source_sha256": config.get("source_sha256"),
                "summary": summary,
                "model_id": self.model_id,
                "prompt_version": PERSONA_SUMMARY_PROMPT_VERSION,
            }
            await self._append_record(summary_path, record)
            summary_keys.add(key)
            await report_call_completed("summary")

        summary_errors: list[Exception] = []
        if summaries_required:
            emit_progress(
                {
                    "type": "simulation_stage",
                    "stage": "PersonaSummary",
                    "status": "started",
                    "total": scheduled_summary_calls,
                }
            )
            if scheduled_summary_calls == 0:
                emit_progress(
                    {
                        "type": "simulation_progress",
                        "stage": "PersonaSummary",
                        "completed": 0,
                        "total": 0,
                    }
                )
            summary_results = await asyncio.gather(
                *(
                    run_summary_job(respondent_ref)
                    for respondent_ref in respondent_refs
                ),
                return_exceptions=True,
            )
            summary_errors = [
                result
                for result in summary_results
                if isinstance(result, Exception)
            ]
            emit_progress(
                {
                    "type": "simulation_stage",
                    "stage": "PersonaSummary",
                    "status": (
                        "failed" if summary_errors else "completed"
                    ),
                    "total": scheduled_summary_calls,
                }
            )

        summary_by_id = {
            record["respondent_id"]: record["summary"]
            for record in _load_jsonl_records(summary_path)
            if isinstance(record.get("respondent_id"), str)
            and isinstance(record.get("summary"), str)
        }
        persona_prompts: dict[tuple[str, str], str] = {}
        if not summary_errors:
            for respondent_id, profile in profile_by_id.items():
                for persona_mode in persona_modes:
                    persona_prompts[(respondent_id, persona_mode)] = (
                        build_persona_prompt(
                            profile,
                            persona_mode=persona_mode,
                            personality_summary=summary_by_id.get(
                                respondent_id
                            ),
                        )
                    )

        def validate_sjt(
            result: Mapping[str, Any],
            *,
            allowed_option_ids: set[str],
        ) -> str:
            selected = result.get("selected_option_id")
            if selected not in allowed_option_ids:
                raise ValueError(
                    f"模型选择了无效 SJT 选项 {selected!r}"
                )
            return str(selected)

        async def run_sjt_job(
            respondent_ref: Mapping[str, Any],
            item: Mapping[str, Any],
            persona_mode: str,
        ) -> None:
            key = (
                respondent_ref["respondent_id"],
                persona_mode,
                item.get("item_id"),
                item.get("item_version"),
            )
            if key in sjt_keys:
                return
            option_ids = {
                option["option_id"]
                for option in item["response_options"]
            }
            selected_option_id = await _invoke_with_retry(
                self.sjt_model,
                build_sjt_messages(
                    persona_prompts[
                        (respondent_ref["respondent_id"], persona_mode)
                    ],
                    item,
                ),
                semaphore=self.semaphore,
                validator=lambda result: validate_sjt(
                    result,
                    allowed_option_ids=option_ids,
                ),
                max_retries=self.max_retries,
                retry_delay_seconds=self.retry_delay_seconds,
                request_timeout_seconds=self.request_timeout_seconds,
                job_label=(
                    f"SJT {respondent_ref['respondent_id']} / "
                    f"{persona_mode} / {item.get('item_id')}"
                ),
            )
            record = {
                "record_type": "sjt_response",
                "run_id": state["run_id"],
                "respondent_id": respondent_ref["respondent_id"],
                "persona_mode": persona_mode,
                "item_bank_id": state["item_bank_id"],
                "item_bank_version": state["item_bank_version"],
                "item_bank_fingerprint": state.get(
                    "item_bank_fingerprint"
                ),
                "item_id": item.get("item_id"),
                "item_version": item.get("item_version"),
                "selected_option_id": selected_option_id,
                "response_mode": "single_item_independent_call",
                "model_id": self.model_id,
                "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
            }
            await self._append_record(sjt_path, record)
            sjt_keys.add(key)
            await report_call_completed("sjt")

        def validate_neo(result: Mapping[str, Any]) -> list[int]:
            ratings = result.get("ratings")
            if not isinstance(ratings, list) or len(ratings) != 12:
                raise ValueError("Neo-FFI 批次必须返回12个评分")
            if any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
                or value > 5
                for value in ratings
            ):
                raise ValueError("Neo-FFI 评分必须是1到5之间的整数")
            return ratings

        async def run_neo_job(
            respondent_ref: Mapping[str, Any],
            dimension: Mapping[str, Any],
            persona_mode: str,
        ) -> None:
            items = dimension["items"]
            pending_items = [
                item
                for item in items
                if (
                    respondent_ref["respondent_id"],
                    persona_mode,
                    item["item_id"],
                )
                not in neo_keys
            ]
            if not pending_items:
                return
            if len(pending_items) != 12:
                raise ValueError(
                    "Neo-FFI 同一维度只允许整批恢复；检测到不完整批次"
                )
            ratings = await _invoke_with_retry(
                self.neo_ffi_model,
                build_neo_ffi_messages(
                    persona_prompts[
                        (respondent_ref["respondent_id"], persona_mode)
                    ],
                    items,
                ),
                semaphore=self.semaphore,
                validator=validate_neo,
                max_retries=self.max_retries,
                retry_delay_seconds=self.retry_delay_seconds,
                request_timeout_seconds=self.request_timeout_seconds,
                job_label=(
                    f"Neo-FFI {respondent_ref['respondent_id']} / "
                    f"{persona_mode} / {dimension['dimension_code']}"
                ),
            )
            records = []
            for item, rating in zip(items, ratings):
                records.append(
                    {
                        "record_type": "neo_ffi_response",
                        "run_id": state["run_id"],
                        "respondent_id": respondent_ref["respondent_id"],
                        "persona_mode": persona_mode,
                        "dimension_code": dimension["dimension_code"],
                        "item_id": item["item_id"],
                        "raw_response": rating,
                        "scoring_direction": item["scoring_direction"],
                        "response_mode": "dimension_batch_12",
                        "model_id": self.model_id,
                        "prompt_version": (
                            VIRTUAL_RESPONSE_PROMPT_VERSION
                        ),
                    }
                )
            await self._append_records(neo_path, records)
            for item in items:
                neo_keys.add(
                    (
                        respondent_ref["respondent_id"],
                        persona_mode,
                        item["item_id"],
                    )
                )
            await report_call_completed("neo")

        if not summary_errors:
            emit_progress(
                {
                    "type": "simulation_stage",
                    "stage": "SJT",
                    "status": "started",
                    "total": scheduled_sjt_calls,
                }
            )
        if not summary_errors and scheduled_sjt_calls == 0:
            emit_progress(
                {
                    "type": "simulation_progress",
                    "stage": "SJT",
                    "completed": 0,
                    "total": 0,
                }
            )
        sjt_jobs = (
            [
                run_sjt_job(respondent_ref, item, persona_mode)
                for respondent_ref in respondent_refs
                for persona_mode in persona_modes
                for item in sjt_items
            ]
            if not summary_errors
            else []
        )
        sjt_results = await asyncio.gather(
            *sjt_jobs,
            return_exceptions=True,
        )
        sjt_errors = [
            result for result in sjt_results if isinstance(result, Exception)
        ]
        if not summary_errors:
            emit_progress(
                {
                    "type": "simulation_stage",
                    "stage": "SJT",
                    "status": "failed" if sjt_errors else "completed",
                    "total": scheduled_sjt_calls,
                }
            )

        neo_errors: list[Exception] = []
        if not sjt_errors:
            emit_progress(
                {
                    "type": "simulation_stage",
                    "stage": "Neo-FFI",
                    "status": "started",
                    "total": scheduled_neo_calls,
                }
            )
            if scheduled_neo_calls == 0:
                emit_progress(
                    {
                        "type": "simulation_progress",
                        "stage": "Neo-FFI",
                        "completed": 0,
                        "total": 0,
                    }
                )
            neo_jobs = [
                run_neo_job(respondent_ref, dimension, persona_mode)
                for respondent_ref in respondent_refs
                for persona_mode in persona_modes
                for dimension in neo_dimensions
            ]
            neo_results = await asyncio.gather(
                *neo_jobs,
                return_exceptions=True,
            )
            neo_errors = [
                result
                for result in neo_results
                if isinstance(result, Exception)
            ]
            emit_progress(
                {
                    "type": "simulation_stage",
                    "stage": "Neo-FFI",
                    "status": "failed" if neo_errors else "completed",
                    "total": scheduled_neo_calls,
                }
            )

        errors = [*summary_errors, *sjt_errors, *neo_errors]
        completed = (
            not errors
            and len(summary_keys) == expected_summary_records
            and len(sjt_keys) == expected_sjt_records
            and len(neo_keys) == expected_neo_records
        )
        manifest.update(
            {
                "status": "completed" if completed else "failed",
                "completed_persona_summary_records": len(summary_keys),
                "completed_sjt_records": len(sjt_keys),
                "completed_neo_ffi_records": len(neo_keys),
                "resumed_persona_summary_records": resumed_summary_count,
                "resumed_sjt_records": resumed_sjt_count,
                "resumed_neo_ffi_records": resumed_neo_count,
                "finished_at": utc_timestamp(),
                "errors": [str(error) for error in errors[:20]],
            }
        )
        _write_json_atomic(manifest_path, manifest)
        if not completed:
            if errors:
                raise RuntimeError(
                    f"虚拟作答有 {len(errors)} 个任务失败；"
                    f"首个错误：{errors[0]}"
                )
            raise RuntimeError("虚拟作答记录数与预期不一致")

        return {
            "manifest_path": str(manifest_path.resolve()),
            "scoring_snapshot_path": str(scoring_snapshot_path.resolve()),
            "persona_summary_path": str(summary_path.resolve()),
            "sjt_path": str(sjt_path.resolve()),
            "neo_ffi_path": str(neo_path.resolve()),
            "respondent_count": len(respondent_refs),
            "persona_modes": persona_modes,
            "persona_mode_count": len(persona_modes),
            "persona_summary_count": len(summary_keys),
            "sjt_item_count": len(sjt_items),
            "sjt_response_count": len(sjt_keys),
            "neo_ffi_response_count": len(neo_keys),
            "scheduled_persona_summary_api_calls": (
                scheduled_summary_calls
            ),
            "scheduled_sjt_api_calls": (
                scheduled_sjt_calls
            ),
            "scheduled_neo_ffi_api_calls": (
                scheduled_neo_calls
            ),
            "max_concurrency": self.max_concurrency,
            "max_retries": self.max_retries,
            "request_timeout_seconds": self.request_timeout_seconds,
            "resumed_persona_summary_records": resumed_summary_count,
            "resumed_sjt_records": resumed_sjt_count,
            "resumed_neo_ffi_records": resumed_neo_count,
            "reused_persona_summary_records": reuse_summary[
                "reused_persona_summary_records"
            ],
            "reused_sjt_records": reuse_summary[
                "reused_sjt_records"
            ],
            "reused_neo_ffi_records": reuse_summary[
                "reused_neo_ffi_records"
            ],
            "source_manifest_path": reuse_summary[
                "source_manifest_path"
            ],
            "model_id": self.model_id,
            "prompt_version": VIRTUAL_RESPONSE_PROMPT_VERSION,
            "persona_summary_prompt_version": (
                PERSONA_SUMMARY_PROMPT_VERSION
            ),
        }


def resolve_virtual_response_output_dir(
    state: Mapping[str, Any],
    output_root: str | Path,
) -> Path:
    """Isolate responses by frozen-bank identity while supporting old runs."""

    run_dir = Path(output_root) / str(state["run_id"])
    fingerprint = str(state.get("item_bank_fingerprint") or "unknown")
    version = state.get("item_bank_version") or 0
    versioned_dir = run_dir / f"bank-v{version}-{fingerprint[:12]}"
    if versioned_dir.exists():
        return versioned_dir

    legacy_manifest = run_dir / "manifest.json"
    if legacy_manifest.exists():
        try:
            legacy = json.loads(
                legacy_manifest.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError):
            legacy = {}
        if (
            legacy.get("item_bank_id") == state.get("item_bank_id")
            and legacy.get("item_bank_version")
            == state.get("item_bank_version")
            and legacy.get("item_bank_fingerprint")
            == state.get("item_bank_fingerprint")
        ):
            return run_dir
    return versioned_dir


async def run_virtual_response_simulation(
    state: PSJTState,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    base_model: Any | None = None,
    neo_ffi_path: str | Path = DEFAULT_NEO_FFI_PATH,
    retry_delay_seconds: float = 1.0,
    request_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """执行虚拟作答并返回允许提交到 State 的轻量更新。"""

    config = state.get("virtual_sample_config") or {}
    max_concurrency = config.get(
        "max_concurrency",
        DEFAULT_MAX_CONCURRENCY,
    )
    max_retries = config.get("max_retries", DEFAULT_MAX_RETRIES)
    runner = VirtualResponseRunner(
        base_model=base_model,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        request_timeout_seconds=request_timeout_seconds,
    )
    output_dir = resolve_virtual_response_output_dir(state, output_root)
    summary = await runner.run(
        state=state,
        output_dir=output_dir,
        neo_ffi_path=neo_ffi_path,
    )
    return {
        "state_update": {
            "virtual_response_data_ref": summary["manifest_path"],
            "virtual_response_summary": summary,
            "virtual_response_item_bank_id": state["item_bank_id"],
            "virtual_response_item_bank_version": state[
                "item_bank_version"
            ],
        },
        "summary": (
            f"已完成 {summary['respondent_count']} 名虚拟被试、"
            f"{summary['persona_mode_count']} 种人格提示模式的"
            f" {summary['sjt_response_count']} 条SJT作答和"
            f" {summary['neo_ffi_response_count']} 条Neo-FFI作答；"
            f"复用未变题作答 {summary['reused_sjt_records']} 条，"
            f"新增SJT调用 {summary['scheduled_sjt_api_calls']} 次，"
            f"复用Neo-FFI作答 {summary['reused_neo_ffi_records']} 条；"
            f"最大并发 {summary['max_concurrency']}"
        ),
    }
