# Compact Item Model Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop item generation, revision, and regeneration requests from sending the complete workflow state to the language model.

**Architecture:** Add one pure context-projection function in `context_control.py` that returns a fixed whitelist of fields required by the item writer. Use that projection in `execute_item_action_with_repair`; keep the complete state available locally for deterministic validation, checkpointing, and audit.

**Tech Stack:** Python 3.11, LangChain runnables, pytest, existing `PSJTState` dictionaries.

---

### Task 1: Define and verify the item-model state boundary

**Files:**
- Modify: `tests/test_context_control.py`
- Modify: `sjt_system/context_control.py`

- [ ] **Step 1: Write the failing projection test**

Import `build_item_model_state`, create a state containing all six allowed fields plus
`execution_history`, `virtual_respondents`, `item_pool`, and `item_history`, then assert
that the result contains exactly:

```python
{
    "current_item",
    "current_item_specification",
    "current_blueprint_cell",
    "test_specification",
    "active_psychometric_diagnostic",
    "user_feedback",
}
```

Also assert the allowed values are preserved by identity or equality.

- [ ] **Step 2: Run the projection test and verify RED**

Run:

```powershell
python -m pytest -q tests\test_context_control.py::test_item_model_state_excludes_workflow_history
```

Expected: collection failure because `build_item_model_state` does not exist.

- [ ] **Step 3: Implement the fixed whitelist projection**

Add to `sjt_system/context_control.py`:

```python
ITEM_MODEL_STATE_FIELDS = (
    "current_item",
    "current_item_specification",
    "current_blueprint_cell",
    "test_specification",
    "active_psychometric_diagnostic",
    "user_feedback",
)


def build_item_model_state(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        field: state.get(field)
        for field in ITEM_MODEL_STATE_FIELDS
    }
```

- [ ] **Step 4: Run the projection test and verify GREEN**

Run:

```powershell
python -m pytest -q tests\test_context_control.py::test_item_model_state_excludes_workflow_history
```

Expected: `1 passed`.

### Task 2: Use the compact state in real item-model requests

**Files:**
- Modify: `tests/test_item_output_repair.py`
- Modify: `sjt_system/execute.py`

- [ ] **Step 1: Write the failing request-boundary test**

Extend the item-action test state with large sentinel values in `execution_history`,
`virtual_respondents`, `item_pool`, and `item_history`. Invoke the real
`execute_agent` path with `SequenceAgent`, then assert:

```python
assert set(agent.inputs[0]["state"]) == {
    "current_item",
    "current_item_specification",
    "current_blueprint_cell",
    "test_specification",
    "active_psychometric_diagnostic",
    "user_feedback",
}
assert "execution_history" not in agent.inputs[0]["state"]
```

Make the first candidate invalid and the second candidate valid, then repeat the
exclusion assertion for `agent.inputs[1]["state"]` to cover output-repair retries.

- [ ] **Step 2: Run the request test and verify RED**

Run:

```powershell
python -m pytest -q tests\test_item_output_repair.py::test_item_requests_use_compact_state_during_repairs
```

Expected: assertion failure showing `execution_history` and other full-state fields
are still present.

- [ ] **Step 3: Route requests through the projection**

Import `build_item_model_state` in `sjt_system/execute.py` and replace:

```python
"state": state,
```

with:

```python
"state": build_item_model_state(state),
```

Do not change local calls to `validate_item_agent_update`; they must continue using
the complete state.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest -q tests\test_context_control.py tests\test_item_output_repair.py
```

Expected: all tests pass.

### Task 3: Verify request-size reduction and regression safety

**Files:**
- No production changes expected.

- [ ] **Step 1: Measure the latest failed checkpoint using the production projection**

Load the newest checkpoint, build the same item request dictionary used by
`execute_item_action_with_repair`, and compare its Python string length with the
full-state request. Confirm the compact request does not contain the
`execution_history` sentinel and is below 50,000 characters for the observed
checkpoint.

- [ ] **Step 2: Run syntax and full regression checks**

Run:

```powershell
python -m py_compile sjt_system\context_control.py sjt_system\execute.py
python -m pytest -q
```

Expected: compilation succeeds and the full suite reports zero failures.

## Repository note

The workspace currently does not expose a usable Git repository (`git rev-parse`
fails), so this plan does not include commit commands. No destructive Git recovery
will be attempted as part of this change.
