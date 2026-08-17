# Workflow Reliability Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the confirmed structured-output, oversized-context, model-owned identity, trace-growth, retry-observability, and environment-reproducibility failures from the PSJT workflow.

**Architecture:** Keep the complete `PSJTState` inside the deterministic workflow and checkpoint layer, but project explicit bounded contexts at every model boundary. Treat model output as untrusted text: parse, validate, normalize, and repair it through one bounded output-repair layer; then deterministically restore program-owned identity fields before business validation.

**Tech Stack:** Python 3.11, LangGraph 1.2, LangChain Core 1.4, LangChain OpenAI 1.3, Pydantic 2.13, pytest.

---

## Phase 1: Stop current production failures

### Task 1: Repair malformed JSON output and return normalized values

**Files:**
- Modify: `tests/test_model_retry.py`
- Modify: `tests/test_structured_output.py`
- Modify: `sjt_system/model_retry.py`
- Modify: `sjt_system/agent/API.py`

- [ ] **Step 1: Add a failing malformed-JSON repair test**

Add a test whose fake model returns an `AIMessage` containing:

```json
{"constraints":["无"好/坏"道德标签"]}
```

on the first call and valid JSON on the second call. Invoke the real
`with_compatible_structured_output` plus `ainvoke_model_with_schema_repair`
pipeline and assert:

```python
assert result == {"selected_option_id": "A"}
assert len(received) == 2
assert "JSON 解析失败" in received[1]["input_data"]["validation_feedback"]
assert received[1]["input_data"]["previous_invalid_candidate"] == invalid_json
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest -q tests\test_model_retry.py::test_invalid_json_is_repaired_with_raw_output_feedback
```

Expected: `OutputParserException` escapes after the first call.

- [ ] **Step 3: Add a failing normalization test**

Define a local `TypedDict` with `count: int`; return `{"count":"4"}` from a
plain-JSON fake model and assert the wrapper returns:

```python
{"count": 4}
```

- [ ] **Step 4: Verify the normalization test is RED**

Run:

```powershell
python -m pytest -q tests\test_structured_output.py::test_plain_json_returns_pydantic_normalized_values
```

Expected: the actual value remains the string `"4"`.

- [ ] **Step 5: Implement one output-repair exception boundary**

In `sjt_system/model_retry.py`, import:

```python
from langchain_core.exceptions import OutputParserException
```

Catch both `LocalJSONSchemaError` and `OutputParserException`. For parser
exceptions, use `exc.llm_output` as the candidate and send a concise feedback
message:

```python
if isinstance(exc, LocalJSONSchemaError):
    candidate = exc.candidate
    feedback = str(exc)
else:
    candidate = exc.llm_output
    feedback = (
        "JSON 解析失败：输出不是合法 JSON。"
        "字符串内部的双引号必须写成 \\\"，"
        "也可以改用中文引号。"
    )
```

Keep the current maximum of three output-repair attempts and the existing
`output_repair` progress event.

- [ ] **Step 6: Return normalized dictionaries**

In `sjt_system/agent/API.py`, replace the discarded validation result with:

```python
validated = output_adapter.validate_python(value)
model_dump = getattr(validated, "model_dump", None)
if callable(model_dump):
    return model_dump()
return validated
```

This preserves dictionaries for workflow code while applying Pydantic
normalization and `extra="forbid"` behavior.

- [ ] **Step 7: Verify GREEN**

Run:

```powershell
python -m pytest -q tests\test_model_retry.py tests\test_structured_output.py
```

Expected: all structured-output and retry tests pass.

### Task 2: Make item identity and version program-owned

**Files:**
- Modify: `tests/test_item_output_repair.py`
- Modify: `tests/test_item_flow.py`
- Modify: `sjt_system/item_flow.py`
- Modify: `sjt_system/execute.py`

- [ ] **Step 1: Add a failing canonicalization unit test**

Add `canonicalize_item_agent_update` tests showing that a revision candidate
which returns the wrong item ID, cell ID, dimension ID, context category, and
version is normalized to:

```python
{
    "item_id": previous_item["item_id"],
    "blueprint_cell_id": current_blueprint_cell["cell_id"],
    "target_dimension_id": current_blueprint_cell["dimension_id"],
    "context_category": current_item_specification["context_category"],
    "version": previous_item["version"] + 1,
}
```

The function must copy the candidate before changing it.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest -q tests\test_item_flow.py::test_item_canonicalization_restores_program_owned_fields
```

Expected: import failure because `canonicalize_item_agent_update` does not exist.

- [ ] **Step 3: Implement deterministic canonicalization**

Add to `sjt_system/item_flow.py`:

```python
def canonicalize_item_agent_update(
    action: str,
    update: dict[str, Any],
    *,
    blueprint_cell: dict[str, Any] | None,
    item_specification: dict[str, Any] | None,
    previous_item: dict[str, Any] | None,
) -> dict[str, Any]:
    canonical = deepcopy(update)
    item = canonical.get("current_item")
    if not isinstance(item, dict):
        return canonical
    if blueprint_cell:
        item["blueprint_cell_id"] = blueprint_cell.get("cell_id")
        item["target_dimension_id"] = blueprint_cell.get("dimension_id")
    if item_specification:
        item["context_category"] = item_specification.get(
            "context_category"
        )
    if action in {"revise_item", "regenerate_item"} and previous_item:
        item["item_id"] = previous_item.get("item_id")
        item["version"] = previous_item.get("version", 0) + 1
    return canonical
```

- [ ] **Step 4: Add a failing integration test**

Make `SequenceAgent` return a revision candidate with `version=1` while the
previous item is also version 1. Assert `execute_agent` succeeds on one model
call and returns version 2.

- [ ] **Step 5: Integrate before business validation**

In `execute_item_action_with_repair`, canonicalize `result["state_update"]`
before calling `validate_item_agent_update`, then return the canonical update.
Do not canonicalize scenario text, response options, scoring keys, or rationale.

- [ ] **Step 6: Verify GREEN**

Run:

```powershell
python -m pytest -q tests\test_item_flow.py tests\test_item_output_repair.py
```

Expected: the version regression and existing item repair tests pass.

## Phase 2: Bound every model input

### Task 3: Add explicit Router, construct, requirement, and audit contexts

**Files:**
- Modify: `tests/test_context_control.py`
- Modify: `tests/test_first_item_integration.py`
- Modify: `tests/test_item_bank_freeze.py`
- Modify: `sjt_system/context_control.py`
- Modify: `sjt_system/graph.py`
- Modify: `sjt_system/execute.py`
- Modify: `sjt_system/prompt/router_prompt.py`

- [ ] **Step 1: Add failing projection tests**

Create states containing sentinel values in:

```python
execution_history
virtual_respondents
item_history
psychometric_revision_history
generation_strategy_history
```

Test four pure projections:

```python
build_router_model_state(state)
build_construct_model_state(state)
build_requirement_model_state(state)
build_item_bank_audit_state(state)
```

Assert none contains the sentinel history fields.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest -q tests\test_context_control.py -k "router_model_state or construct_model_state or requirement_model_state or item_bank_audit_state"
```

Expected: imports fail because the projections do not exist.

- [ ] **Step 3: Implement the construct projection**

Return only:

```python
{
    "user_request": state.get("user_request"),
    "test_specification": state.get("test_specification"),
    "quality_constraints": state.get("quality_constraints"),
    "theory_evidence": state.get("theory_evidence") or [],
    "user_feedback": state.get("user_feedback"),
}
```

- [ ] **Step 4: Implement the requirement projection**

Return only the fields documented by `REQUIREMENT_PROMPT`:

```python
{
    "user_request": state.get("user_request"),
    "test_specification": state.get("test_specification"),
    "pending_state_update": state.get("pending_state_update"),
    "specification_sources": state.get("specification_sources") or {},
    "user_feedback": state.get("user_feedback"),
    "confirmed_requirement_fields": (
        state.get("confirmed_requirement_fields") or []
    ),
    "requirement_conversation": (
        state.get("requirement_conversation") or []
    ),
}
```

- [ ] **Step 5: Implement an audit snapshot**

Move `build_item_bank_snapshot` from `graph.py` to `context_control.py` and
reuse it for both approval display and model input. Build the audit state from:

```python
{
    "test_specification": state.get("test_specification"),
    "blueprint": state.get("blueprint"),
    "item_bank_snapshot": build_item_bank_snapshot(state),
    "context_usage": state.get("context_usage") or {},
    "item_pattern_profiles": state.get("item_pattern_profiles") or {},
}
```

- [ ] **Step 6: Implement a Router summary**

Return only route-relevant status and bounded summaries:

```python
{
    "requirements_confirmed": state.get("requirements_confirmed"),
    "test_specification": state.get("test_specification"),
    "theory_search_completed": state.get("theory_search_completed"),
    "construct_model_summary": {
        "exists": state.get("construct_model") is not None,
        "dimension_count": len(
            (state.get("construct_model") or {}).get("dimensions") or []
        ),
    },
    "blueprint_summary": {
        "exists": state.get("blueprint") is not None,
        "cell_count": len(
            (state.get("blueprint") or {}).get("cells") or []
        ),
    },
    "blueprint_progress": state.get("blueprint_progress") or {},
    "current_item": state.get("current_item"),
    "current_review_decision": state.get("current_review_decision"),
    "item_revision_counts": state.get("item_revision_counts") or {},
    "current_item_regeneration_count": (
        state.get("current_item_regeneration_count")
    ),
    "max_item_revision_count": state.get("max_item_revision_count"),
    "max_item_regeneration_count": (
        state.get("max_item_regeneration_count")
    ),
    "item_pool_count": len(state.get("item_pool") or []),
    "item_bank_audit": state.get("item_bank_audit"),
    "supplement_request": state.get("supplement_request"),
    "frozen_item_bank_exists": bool(state.get("frozen_item_bank")),
    "virtual_response_data_ref": state.get(
        "virtual_response_data_ref"
    ),
    "psychometric_results_present": {
        field: state.get(field) is not None
        for field in (
            "item_statistics",
            "test_statistics",
            "factor_results",
            "irt_results",
            "dif_results",
        )
    },
    "selection_results": state.get("selection_results"),
    "assembled_test_exists": state.get("assembled_test") is not None,
    "test_review_result": state.get("test_review_result"),
    "final_outputs_present": {
        field: state.get(field) is not None
        for field in (
            "final_test",
            "item_database_ref",
            "technical_report",
            "virtual_respondent_report",
        )
    },
}
```

Update the Router prompt to state that it receives these summaries rather than
the complete `PSJTState`.

- [ ] **Step 7: Wire all model boundaries**

Use:

```python
{"input_data": build_router_model_state(state)}
```

in `router_node`; use `build_construct_model_state` in construct modeling;
use `build_item_bank_audit_state` in bank audit; and use
`build_requirement_model_state` in the requirement fallback.

- [ ] **Step 8: Add request-boundary integration assertions**

Capture real fake-Agent inputs and assert `execution_history` is absent from
Router and bank-audit model requests.

- [ ] **Step 9: Verify GREEN and measure**

Run:

```powershell
python -m pytest -q tests\test_context_control.py tests\test_first_item_integration.py tests\test_item_bank_freeze.py
```

Then load the latest checkpoint and assert each projected request is below
50,000 characters.

## Phase 3: Control state growth and retry observability

### Task 4: Bound nested trace strings and stored error messages

**Files:**
- Modify: `tests/test_trace.py`
- Modify: `tests/test_run_checkpoint.py`
- Modify: `sjt_system/trace.py`
- Modify: `sjt_system/graph.py`

- [ ] **Step 1: Add a failing nested-string test**

Assert:

```python
summary = summarize_value(
    {"a": {"b": {"c": "x" * 10_000}}}
)
assert len(summary["a"]["b"]["c"]) <= 301
assert summary["a"]["b"]["c"].endswith("…")
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest -q tests\test_trace.py::test_summarize_value_truncates_deep_strings
```

Expected: the deep value remains 10,000 characters.

- [ ] **Step 3: Fix truncation order**

Move the string branch before the depth branch:

```python
if isinstance(value, str):
    if len(value) <= _MAX_STRING_LENGTH:
        return value
    return value[:_MAX_STRING_LENGTH] + "…"

if depth >= _MAX_DEPTH:
    ...
```

- [ ] **Step 4: Add and test bounded error storage**

Add:

```python
_MAX_ERROR_LENGTH = 2000


def summarize_error_message(error: object) -> str:
    message = str(error)
    if len(message) <= _MAX_ERROR_LENGTH:
        return message
    return message[:_MAX_ERROR_LENGTH] + "…"
```

Use it in Router and Execute exception handlers before writing `errors` and
`execution_history`. Keep the exception type or action name in surrounding
fields.

- [ ] **Step 5: Verify GREEN**

Run:

```powershell
python -m pytest -q tests\test_trace.py tests\test_run_checkpoint.py tests\test_app_output.py
```

Expected: nested strings and long model errors are bounded.

### Task 5: Make network attempt counts explicit

**Files:**
- Modify: `tests/test_structured_output.py`
- Modify: `tests/test_model_retry.py`
- Modify: `sjt_system/agent/API.py`
- Modify: `sjt_system/model_retry.py`
- Modify: `README.md`

- [ ] **Step 1: Add a failing model configuration test**

Assert `get_model()` sets:

```python
model.max_retries == 0
```

so the application-level retry layer owns network retry counting.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest -q tests\test_structured_output.py::test_model_disables_hidden_sdk_retries
```

Expected: `max_retries` is `None`.

- [ ] **Step 3: Disable hidden SDK retries**

Pass:

```python
max_retries=0
```

to `ChatOpenAI`. Preserve the existing per-attempt timeout and bounded outer
retry behavior.

- [ ] **Step 4: Clarify progress metadata**

Add `retry_kind` with values `network_timeout` or `output_repair` to progress
events and update application rendering/tests accordingly. Document that
semantic business repair loops are separate from network retries.

- [ ] **Step 5: Verify GREEN**

Run:

```powershell
python -m pytest -q tests\test_model_retry.py tests\test_structured_output.py tests\test_app_output.py
```

Expected: retry counts and categories match emitted events.

## Phase 4: Reproducible runtime verification

### Task 6: Pin runtime and development dependencies

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Modify: `README.md`

- [ ] **Step 1: Add the runtime dependency manifest**

Create:

```text
langgraph==1.2.9
langchain-openai==1.3.5
langchain-core==1.4.9
openai==2.45.0
pydantic==2.13.4
python-dotenv==1.2.2
typing-extensions==4.16.0
numpy==2.4.6
pandas==3.0.5
scipy==1.17.1
requests==2.34.2
tavily-python==0.7.26
```

- [ ] **Step 2: Add development dependencies**

Create:

```text
-r requirements.txt
pytest
pytest-cov
coverage
```

- [ ] **Step 3: Correct README setup and architecture**

Document:

```powershell
.\env\Scripts\python.exe -m pip install -r requirements-dev.txt
.\env\Scripts\python.exe -m pytest -q
.\env\Scripts\python.exe app.py
```

Remove the outdated claim that only two handlers are implemented. Describe the
plain-JSON parse, validation, normalization, repair, and bounded-context flow.

- [ ] **Step 4: Verify the actual virtual environment**

Install only after explicit package-install approval if packages are missing.
Then run:

```powershell
.\env\Scripts\python.exe -m pip check
.\env\Scripts\python.exe -m pytest -q
```

Expected: no dependency conflicts and all tests pass under the same interpreter
used to run the application.

## Final verification

- [ ] Run:

```powershell
python -m py_compile sjt_system\agent\API.py sjt_system\model_retry.py sjt_system\context_control.py sjt_system\item_flow.py sjt_system\execute.py sjt_system\graph.py sjt_system\trace.py
python -m pytest -q
```

- [ ] Re-run the exact malformed blueprint JSON regression.
- [ ] Re-run the exact revision-version regression.
- [ ] Measure Router, construct, audit, and item request sizes from the latest checkpoint.
- [ ] Confirm resumed failed runs do not resend complete execution history.

## Repository note

The workspace contains an empty `.git` directory and is not a usable Git
repository. This plan therefore omits commit and branch commands. It will not
attempt destructive Git recovery.
