# Plain JSON Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop sending unsupported `response_format=json_object` for DeepSeek while preserving structured local validation.

**Architecture:** Add a `plain_json` method to the centralized model-output adapter. The prompt continues to include the generated JSON Schema, while the raw model response is parsed with `JsonOutputParser` and validated locally with `jsonschema`.

**Tech Stack:** Python 3.11, LangChain Core, jsonschema, pytest.

---

### Task 1: Add a tested plain-JSON adapter

**Files:**
- Modify: `tests/test_structured_output.py`
- Modify: `sjt_system/agent/API.py`
- Modify: `sjt_system/agent/agent_factory.py`

- [ ] Write a failing test that DeepSeek defaults to `plain_json`.
- [ ] Write a failing test that fenced or plain JSON text is parsed locally.
- [ ] Write a failing test that a schema-invalid JSON object raises `ValueError`.
- [ ] Add `plain_json` to the supported output methods.
- [ ] Build `model | JsonOutputParser | RunnableLambda(schema_validator)` without calling `with_structured_output`.
- [ ] Append the existing JSON Schema instruction for `plain_json`.
- [ ] Run `tests/test_structured_output.py` and agent contract tests.

### Task 2: Document and verify

**Files:**
- Modify: `README.md`

- [ ] Document that DeepSeek defaults to prompt-driven plain JSON and local validation.
- [ ] Run syntax compilation for the modified modules.
- [ ] Run the complete pytest suite.
