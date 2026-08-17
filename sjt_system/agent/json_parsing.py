"""Parse and conservatively repair model-generated JSON text."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_partial_json


LOGGER = logging.getLogger(__name__)

_CODE_BLOCK_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)```",
    flags=re.IGNORECASE,
)
_QUOTED_PROPERTY_FRAGMENT_RE = re.compile(
    r'([,{]\s*)"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*'
    r'(true|false|null|[+-]?\d+(?:\.\d+)?)"(?=\s*[,}])'
)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _message_text(value: Any) -> str:
    """Extract textual content from a LangChain model message."""

    if isinstance(value, str):
        return value
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)
    raise OutputParserException(
        "模型回复不包含可解析的文本 JSON",
        llm_output=str(content),
    )


def _balanced_json_candidates(text: str) -> list[str]:
    """Extract complete top-level objects or arrays embedded in prose."""

    candidates: list[str] = []
    for start, first in enumerate(text):
        if first not in "[{":
            continue
        stack = ["}" if first == "{" else "]"]
        in_string = False
        escaped = False
        for index in range(start + 1, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char in "}]":
                if not stack or char != stack[-1]:
                    break
                stack.pop()
                if not stack:
                    candidates.append(text[start:index + 1])
                    break
    return candidates


def _candidate_texts(text: str) -> list[tuple[str, str]]:
    """Return distinct JSON-looking candidates and their extraction stage."""

    stripped = text.strip().lstrip("\ufeff")
    values: list[tuple[str, str]] = [("raw", stripped)]
    values.extend(
        ("code_block", match.group(1).strip())
        for match in _CODE_BLOCK_RE.finditer(stripped)
    )
    values.extend(
        ("embedded", candidate.strip())
        for candidate in _balanced_json_candidates(stripped)
    )
    distinct: list[tuple[str, str]] = []
    seen: set[str] = set()
    for stage, candidate in values:
        if candidate and candidate not in seen:
            seen.add(candidate)
            distinct.append((stage, candidate))
    return distinct


def _repair_control_characters(text: str) -> str:
    """Escape raw JSON control characters only when they occur in strings."""

    output: list[str] = []
    in_string = False
    escaped = False
    replacements = {
        "\b": "\\b",
        "\f": "\\f",
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
    }
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            if ord(char) < 0x20:
                output.append(replacements.get(char, ""))
                continue
        elif char == '"':
            in_string = True
        output.append(char)
    return "".join(output)


def _repair_invalid_string_escapes(text: str) -> str:
    """Double invalid backslashes inside strings without changing valid escapes."""

    output: list[str] = []
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if not in_string:
            if char == '"':
                in_string = True
            output.append(char)
            index += 1
            continue
        if char == '"':
            in_string = not in_string
            output.append(char)
            index += 1
            continue
        if char == "\\":
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if next_char == "u":
                unicode_digits = text[index + 2:index + 6]
                if len(unicode_digits) == 4 and all(
                    value in "0123456789abcdefABCDEF"
                    for value in unicode_digits
                ):
                    output.append(text[index:index + 6])
                    index += 6
                    continue
                else:
                    output.append("\\\\")
                    index += 1
                    continue
            elif next_char in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
                output.append(text[index:index + 2])
                index += 2
                continue
            else:
                output.append("\\\\")
                index += 1
                continue
        output.append(char)
        index += 1
    return "".join(output)


def _repair_common_json_errors(text: str) -> str:
    """Apply repairs that do not invent missing domain values."""

    repaired = _QUOTED_PROPERTY_FRAGMENT_RE.sub(
        lambda match: (
            f'{match.group(1)}"{match.group(2)}": {match.group(3)}'
        ),
        text,
    )
    repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
    repaired = _repair_control_characters(repaired)
    repaired = _repair_invalid_string_escapes(repaired)
    return repaired


def parse_model_json_response(value: Any) -> Any:
    """Parse model output with explicit, conservative local repair stages.

    No domain field is synthesized here. Parsed values still pass through the
    caller's Pydantic schema and deterministic business validation.
    """

    raw_text = _message_text(value)
    errors: list[str] = []
    for extraction_stage, candidate in _candidate_texts(raw_text):
        variants = [("strict", candidate)]
        repaired = _repair_common_json_errors(candidate)
        if repaired != candidate:
            variants.append(("repaired", repaired))
        for repair_stage, variant in variants:
            try:
                parsed = json.loads(variant)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"{extraction_stage}/{repair_stage}: "
                    f"{exc.msg} at {exc.pos}"
                )
            else:
                if repair_stage != "strict" or extraction_stage != "raw":
                    LOGGER.info(
                        "Parsed model JSON via %s/%s",
                        extraction_stage,
                        repair_stage,
                    )
                return parsed

        # Preserve the partial-JSON compatibility supplied by LangChain, but
        # only after strict parsing and conservative repairs have failed.
        partial_source = repaired
        if partial_source.lstrip().startswith(("{", "[")):
            try:
                parsed = parse_partial_json(partial_source)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"{extraction_stage}/partial: {exc.msg} at {exc.pos}"
                )
            else:
                if parsed is not None:
                    LOGGER.info(
                        "Parsed model JSON via %s/partial",
                        extraction_stage,
                    )
                    return parsed

    detail = errors[-1] if errors else "没有找到 JSON 对象或数组"
    raise OutputParserException(
        f"本地 JSON 解析与安全修复失败：{detail}",
        llm_output=raw_text,
    )
