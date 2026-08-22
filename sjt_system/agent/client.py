import json
import os
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import TypeAdapter, ValidationError as PydanticValidationError
from sjt_system.agent.json_parsing import parse_model_json_response

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def get_model(
    model_id: str | None = None,
    *,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    thinking_type: str | None = None,
) -> ChatOpenAI:
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Missing API_KEY environment variable.")
    if reasoning_effort is not None:
        reasoning_effort = reasoning_effort.strip().lower()
        if reasoning_effort not in {"low", "medium", "high", "max"}:
            raise ValueError(
                "reasoning_effort must be low, medium, high, or max"
            )
    if thinking_type is not None:
        thinking_type = thinking_type.strip().lower()
        if thinking_type not in {"enabled", "disabled"}:
            raise ValueError("thinking_type must be enabled or disabled")
    if reasoning_effort is not None and thinking_type == "disabled":
        raise ValueError(
            "thinking_type cannot be disabled when reasoning_effort is set"
        )

    # Preserve the historical temperature=1.0 default for ordinary agents.
    # Reasoning agents omit temperature unless the role explicitly sets it,
    # because reasoning endpoints do not consistently accept sampling controls.
    resolved_temperature = (
        None
        if temperature is None and (reasoning_effort or thinking_type)
        else 1.0 if temperature is None else temperature
    )
    if resolved_temperature is not None and (
        not isinstance(resolved_temperature, (int, float))
        or isinstance(resolved_temperature, bool)
        or resolved_temperature < 0
        or resolved_temperature > 2
    ):
        raise ValueError("model temperature must be between 0 and 2")
    return ChatOpenAI(
        model=model_id or os.getenv("MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        base_url=os.getenv("BASE_URL") or None,
        temperature=(
            float(resolved_temperature)
            if resolved_temperature is not None
            else None
        ),
        reasoning_effort=reasoning_effort,
        extra_body=(
            {"thinking": {"type": thinking_type}}
            if thinking_type is not None
            else None
        ),
        timeout=get_model_request_timeout_seconds(),
        max_retries=0,
    )


SUPPORTED_STRUCTURED_OUTPUT_METHODS = {
    "function_calling",
    "json_mode",
    "json_schema",
    "plain_json",
}
DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_MODEL_REQUEST_MAX_ATTEMPTS = 2


class LocalJSONSchemaError(ValueError):
    """A parsed JSON candidate that failed local schema validation."""

    def __init__(
        self,
        message: str,
        *,
        candidate: Any,
        field_path: str,
    ) -> None:
        super().__init__(message)
        self.candidate = candidate
        self.field_path = field_path


def get_model_request_timeout_seconds() -> float:
    """Read and validate the per-request model timeout."""

    raw_value = os.getenv(
        "MODEL_REQUEST_TIMEOUT_SECONDS",
        str(DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS),
    )
    try:
        timeout_seconds = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            "MODEL_REQUEST_TIMEOUT_SECONDS must be a number"
        ) from exc
    if timeout_seconds <= 0 or timeout_seconds > 1800:
        raise ValueError(
            "MODEL_REQUEST_TIMEOUT_SECONDS must be greater than 0 "
            "and no greater than 1800"
        )
    return timeout_seconds


def get_model_request_max_attempts() -> int:
    """Read and validate the total attempts for one model request."""

    raw_value = os.getenv(
        "MODEL_REQUEST_MAX_ATTEMPTS",
        str(DEFAULT_MODEL_REQUEST_MAX_ATTEMPTS),
    )
    try:
        max_attempts = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "MODEL_REQUEST_MAX_ATTEMPTS must be an integer"
        ) from exc
    if max_attempts < 1 or max_attempts > 5:
        raise ValueError(
            "MODEL_REQUEST_MAX_ATTEMPTS must be between 1 and 5"
        )
    return max_attempts


def resolve_structured_output_method(model: Any) -> str:
    """Select a structured-output method supported by the configured model."""

    configured = os.getenv("STRUCTURED_OUTPUT_METHOD", "").strip().lower()
    if configured:
        if configured not in SUPPORTED_STRUCTURED_OUTPUT_METHODS:
            allowed = ", ".join(sorted(SUPPORTED_STRUCTURED_OUTPUT_METHODS))
            raise ValueError(
                "STRUCTURED_OUTPUT_METHOD must be one of: "
                f"{allowed}"
            )
        return configured

    model_id = str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or ""
    ).lower()
    if "deepseek" in model_id:
        return "plain_json"
    return "json_schema"


def build_json_output_instruction(output_type: type) -> str:
    """Build the schema instruction required when the API uses JSON mode."""

    parameters = TypeAdapter(output_type).json_schema()
    serialized_schema = json.dumps(
        parameters,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "Return exactly one valid JSON object and no Markdown or explanatory "
        "text. The JSON object must match this JSON Schema: "
        f"{serialized_schema}"
    )


def with_compatible_structured_output(
    model: Any,
    output_type: type,
) -> tuple[Any, str]:
    """Wrap a model for structured output and return the selected method."""

    method = resolve_structured_output_method(model)
    if method == "plain_json":
        output_adapter = TypeAdapter(output_type)

        def validate_local_json(value: Any) -> Any:
            try:
                validated = output_adapter.validate_python(value)
            except PydanticValidationError as exc:
                issue = exc.errors(include_url=False)[0]
                field_path = "$"
                for part in issue.get("loc", ()):
                    if isinstance(part, int):
                        field_path += f"[{part}]"
                    else:
                        field_path += f".{part}"
                raise LocalJSONSchemaError(
                    "本地 JSON Schema 校验失败"
                    f"（字段 {field_path}）：{issue.get('msg', str(exc))}",
                    candidate=value,
                    field_path=field_path,
                ) from exc
            model_dump = getattr(validated, "model_dump", None)
            if callable(model_dump):
                return model_dump()
            return validated

        runnable = (
            model
            | RunnableLambda(parse_model_json_response)
            | RunnableLambda(validate_local_json)
        )
        return runnable, method
    runnable = model.with_structured_output(
        output_type,
        method=method,
    )
    return runnable, method
