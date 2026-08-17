# Balanced Item Review and Difficulty Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single item reviewer with independent construct and content reviewers, deterministically aggregate their reports with cross-item findings, and remove unsupported author-assigned difficulty from the PSJT blueprint.

**Architecture:** Two structured-output reviewer Agents run concurrently with isolated inputs. A pure aggregation function validates their reports, merges deterministic cross-item findings, and emits the existing workflow-level review fields so revision and regeneration routing remain stable. Difficulty removal is a schema simplification across State, blueprint construction, generation slots, prompts, validators, and fixtures.

**Tech Stack:** Python 3.11, TypedDict, LangChain structured output, LangGraph, asyncio, pytest.

**Repository note:** The current workspace is not recognized as a Git repository, so commit commands cannot be executed until repository metadata is restored.

---

### Task 1: Remove `target_difficulty` with a failing regression test

**Files:**
- Modify: `tests/test_blueprint_flow.py`
- Modify: `tests/test_context_control.py`
- Modify: `sjt_system/state.py`
- Modify: `sjt_system/blueprint_flow.py`
- Modify: `sjt_system/context_control.py`
- Modify: `sjt_system/prompt/blueprint_prompt.py`
- Modify: `tests/test_first_item_integration.py`
- Modify: `tests/test_app_output.py`

- [ ] **Step 1: Write failing omission assertions**

Add to `tests/test_blueprint_flow.py`:

```python
def test_blueprint_does_not_assign_author_difficulty():
    skeleton = build_blueprint_skeleton(
        _specification(), _construct_model(), "run-12345678"
    )
    assert all("target_difficulty" not in cell for cell in skeleton["cells"])
```

Add to `tests/test_context_control.py` after building specifications:

```python
assert all("target_difficulty" not in spec for spec in specifications)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
python -m pytest tests\test_blueprint_flow.py::test_blueprint_does_not_assign_author_difficulty tests\test_context_control.py::test_item_slots_obey_quotas_and_diversify_each_cell -q
```

Expected: failure because blueprint cells and item specifications currently
contain `target_difficulty`.

- [ ] **Step 3: Remove the field from production schemas and behavior**

Delete `target_difficulty` from `BlueprintCell` and `ItemSpecification` in
`sjt_system/state.py`.

Delete this entry from `build_blueprint_skeleton()` in
`sjt_system/blueprint_flow.py`:

```python
"target_difficulty": "medium",
```

Delete both `target_difficulty` validation blocks from
`validate_blueprint_cell_candidate()` and `validate_blueprint()`.

Delete this entry from `build_item_specifications()` in
`sjt_system/context_control.py`:

```python
"target_difficulty": cell["target_difficulty"],
```

Delete the `target_difficulty` instruction from
`sjt_system/prompt/blueprint_prompt.py`.

- [ ] **Step 4: Update fixtures to the new schema**

Remove `"target_difficulty": "medium"` from blueprint-cell fixtures in:

```text
tests/test_blueprint_flow.py
tests/test_context_control.py
tests/test_first_item_integration.py
tests/test_app_output.py
```

- [ ] **Step 5: Run all blueprint and context-control tests**

Run:

```powershell
python -m pytest tests\test_blueprint_flow.py tests\test_context_control.py tests\test_first_item_integration.py tests\test_app_output.py -q
```

Expected: all selected tests pass.

### Task 2: Define independent expert-review contracts

**Files:**
- Modify: `tests/test_item_contract.py`
- Modify: `sjt_system/state.py`
- Modify: `sjt_system/item_flow.py`

- [ ] **Step 1: Write failing expert-contract tests**

Add tests that construct:

```python
construct_result = {
    "reviewer_type": "construct",
    "decision": "PASS",
    "repair_scope": "none",
    "findings": [{
        "criterion": "construct_alignment",
        "severity": "info",
        "evidence": "四个选项沿计划执行程度递增",
        "finding": "与目标维度一致",
        "recommendation": "保持",
    }],
    "summary": "构念审查通过",
}
```

Verify that a `PASS` with a blocking finding raises `ValueError`, and verify
that a `REJECT` without `repair_scope="rewrite"` raises `ValueError`.

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```powershell
python -m pytest tests\test_item_contract.py -q
```

Expected: failure because expert-review validation does not exist.

- [ ] **Step 3: Add expert-review TypedDicts**

Add to `sjt_system/state.py`:

```python
class ExpertReviewFinding(TypedDict):
    criterion: str
    severity: Literal["info", "warning", "blocking"]
    evidence: str
    finding: str
    recommendation: str


class ExpertItemReviewResult(TypedDict):
    reviewer_type: Literal["construct", "content"]
    decision: Literal["PASS", "REVISE", "REJECT"]
    repair_scope: Literal["none", "local", "rewrite"]
    findings: list[ExpertReviewFinding]
    summary: str
```

Add `current_expert_review_results:
list[ExpertItemReviewResult] | None` to `PSJTState` and initialize it to `None`.

- [ ] **Step 4: Add strict expert validation**

Add `validate_expert_review_result()` to `sjt_system/item_flow.py`. It must
enforce non-empty evidence and recommendations, valid reviewer type and
severity, no blocking finding for `PASS`, at least one warning or blocking
finding for `REVISE`, and blocking plus rewrite scope for `REJECT`.

- [ ] **Step 5: Run contract tests**

Run:

```powershell
python -m pytest tests\test_item_contract.py -q
```

Expected: all tests pass.

### Task 3: Add separate reviewer prompts and Agents

**Files:**
- Modify: `sjt_system/prompt/prompt.py`
- Modify: `sjt_system/prompt/__init__.py`
- Modify: `sjt_system/agent/agent_factory.py`
- Modify: `sjt_system/agent/__init__.py`

- [ ] **Step 1: Add a registration test**

Create a test in `tests/test_item_contract.py` that imports
`construct_review_agent` and `content_review_agent` and asserts they are
distinct objects.

- [ ] **Step 2: Run the import test and verify failure**

Run:

```powershell
python -m pytest tests\test_item_contract.py -q
```

Expected: import failure because the Agents are not registered.

- [ ] **Step 3: Refine and export specialist prompts**

Keep `CONSTRUCT_REVIEWER_PROMPT` focused on construct alignment, activation,
unidimensionality, contamination, scoring, and assigned-context compliance.
Keep `CONTENT_REVIEWER_PROMPT` focused on clarity, realism, option quality,
social desirability, answer cues, language, fairness, and supplied similarity
evidence. Both prompts must require `ExpertItemReviewResult` and prohibit
reviewing the other specialist's criteria.

Export both constants from `sjt_system/prompt/__init__.py`.

- [ ] **Step 4: Register two structured-output Agents**

In `sjt_system/agent/agent_factory.py`, replace `review_agent` with:

```python
construct_review_agent = create_agent(
    system_prompt=CONSTRUCT_REVIEWER_PROMPT,
    output_type=ExpertItemReviewResult,
)
content_review_agent = create_agent(
    system_prompt=CONTENT_REVIEWER_PROMPT,
    output_type=ExpertItemReviewResult,
)
```

Export both from `sjt_system/agent/__init__.py`. Remove the misleading
single-reviewer export.

- [ ] **Step 5: Run the registration test**

Run:

```powershell
python -m pytest tests\test_item_contract.py -q
```

Expected: all tests pass.

### Task 4: Build isolated review inputs and deterministic aggregation

**Files:**
- Create: `tests/test_review_aggregation.py`
- Modify: `sjt_system/context_control.py`
- Modify: `sjt_system/item_flow.py`

- [ ] **Step 1: Write failing aggregation tests**

Cover these cases:

```text
construct PASS + content PASS + no deterministic findings -> PASS
construct REVISE + content PASS -> REVISE
construct PASS + content REVISE -> REVISE
validated expert REJECT -> REJECT
deterministic cross-item blocking duplicate -> REJECT
deterministic warning -> REVISE
```

Also assert that construct input contains the construct dimension but not
comparison candidates, and content input contains comparison candidates but
not behavioral anchors.

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests\test_review_aggregation.py -q
```

Expected: failure because input builders and aggregation do not exist.

- [ ] **Step 3: Add isolated input builders**

Add to `sjt_system/context_control.py`:

```python
def build_construct_review_context(state: Mapping[str, Any]) -> dict[str, Any]:
    ...


def build_content_review_context(state: Mapping[str, Any]) -> dict[str, Any]:
    ...
```

The construct builder selects only the matching construct dimension and
behavioral anchors. The content builder includes test specification,
blueprint constraints, deterministic findings, and ranked similar profiles.

- [ ] **Step 4: Add pure aggregation**

Add `aggregate_item_reviews()` to `sjt_system/item_flow.py`. Validate both
expert reports, require exactly one construct and one content result, merge
their findings with deterministic findings, and return:

```python
{
    "current_expert_review_results": expert_results,
    "current_review_results": merged_findings,
    "current_review_decision": decision,
}
```

Classify a deterministic blocking `cross_item_duplication` finding as
`REJECT`; classify deterministic warnings as `REVISE`.

- [ ] **Step 5: Run aggregation tests**

Run:

```powershell
python -m pytest tests\test_review_aggregation.py -q
```

Expected: all tests pass.

### Task 5: Execute both reviewers concurrently

**Files:**
- Modify: `tests/test_first_item_integration.py`
- Modify: `sjt_system/execute.py`

- [ ] **Step 1: Write a failing concurrent-review integration test**

Monkeypatch both reviewer Agents with async fakes that record their inputs.
Call the review execution path and assert:

```python
assert result["state_update"]["current_review_decision"] == "PASS"
assert len(result["state_update"]["current_expert_review_results"]) == 2
assert "comparison_candidates" not in construct_input
assert "behavioral_anchors" not in content_input
```

- [ ] **Step 2: Run the focused integration test and verify failure**

Run:

```powershell
python -m pytest tests\test_first_item_integration.py -q
```

Expected: failure because review execution still calls one Agent.

- [ ] **Step 3: Implement the specialized review executor**

In `sjt_system/execute.py`, remove `review_item` from `AGENT_MAP` and add
`execute_item_review(state)`. Use `asyncio.gather()` to invoke
`construct_review_agent` and `content_review_agent` with their isolated
contexts, compute deterministic cross-item findings once, and call
`aggregate_item_reviews()`.

Route `action == "review_item"` to `execute_item_review()` before the generic
Agent lookup.

- [ ] **Step 4: Run integration tests**

Run:

```powershell
python -m pytest tests\test_first_item_integration.py -q
```

Expected: all tests pass.

### Task 6: Preserve expert reports through validation and history

**Files:**
- Modify: `tests/test_item_flow.py`
- Modify: `sjt_system/item_flow.py`

- [ ] **Step 1: Write failing lifecycle tests**

Assert that:

- `review_item` may update `current_expert_review_results`;
- `_review_history()` stores expert reports;
- accepting or abandoning an item clears working expert reports;
- revising or regenerating clears the prior aggregate and expert reports.

- [ ] **Step 2: Run lifecycle tests and verify failure**

Run:

```powershell
python -m pytest tests\test_item_flow.py -q
```

Expected: failures because expert reports are not yet workflow-owned.

- [ ] **Step 3: Update workflow ownership and cleanup**

Allow `current_expert_review_results` only for `review_item`. Include it in
review history. Clear it in `record_committed_item_output()`,
`build_accept_item_update()`, and `build_abandon_item_update()`.

- [ ] **Step 4: Run lifecycle tests**

Run:

```powershell
python -m pytest tests\test_item_flow.py -q
```

Expected: all tests pass.

### Task 7: Full verification

**Files:**
- Verify all modified production and test files.

- [ ] **Step 1: Compile modified modules**

Run:

```powershell
python -m py_compile sjt_system\state.py sjt_system\blueprint_flow.py sjt_system\context_control.py sjt_system\item_flow.py sjt_system\execute.py sjt_system\agent\agent_factory.py
```

Expected: exit code 0 with no output.

- [ ] **Step 2: Run the full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Search for stale production references**

Run:

```powershell
rg -n "target_difficulty|\breview_agent\b" sjt_system tests
```

Expected: no stale production references; test names may mention rejection of
legacy fields only if such a regression test is intentionally retained.

- [ ] **Step 4: Confirm behavior**

Verify from test evidence that each item receives two independent expert
reports, only the aggregate review requires human confirmation, any non-pass
result returns to revision or regeneration, and no generated blueprint or item
specification contains author-assigned difficulty.
