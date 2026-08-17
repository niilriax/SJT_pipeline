# Timeout Retry and Run Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent transient model timeouts and citation-label drift from terminating an unrecoverable PSJT development run.

**Architecture:** Use `source_id` as the authoritative theory-evidence key and canonicalize display titles in program code. Route all ordinary LLM calls through one bounded timeout-retry helper. Persist app-level JSON snapshots after graph updates and restart a failed run from its committed business state through the Router.

**Tech Stack:** Python 3.11, asyncio, LangChain/LangGraph, pytest, standard-library JSON and pathlib.

**Repository note:** `git rev-parse --show-toplevel` currently reports that the workspace is not a Git repository. Worktree creation and per-task commits are therefore unavailable; verification will use file diffs and automated tests.

---

### Task 1: Canonicalize construct-model source labels

**Files:**
- Modify: `sjt_system/theory_search.py`
- Modify: `tests/test_theory_search.py`

- [ ] **Step 1: Replace the obsolete rejection test with a failing canonicalization test**

Add an import for `canonicalize_construct_model_sources` and replace
`test_construct_model_cannot_invent_citation_label` with:

```python
def test_construct_model_canonicalizes_citation_label_from_source_id():
    evidence = [{"source_id": "src-valid", "title": "Retrieved source title"}]
    update = _construct_model_update(
        source_id="src-valid",
        citation_label="Model-rephrased title",
    )

    canonical = canonicalize_construct_model_sources(update, evidence)

    assert (
        canonical["construct_model"]["source_references"][0]["citation_label"]
        == "Retrieved source title"
    )
    validate_construct_model_sources(canonical, evidence)
    assert (
        update["construct_model"]["source_references"][0]["citation_label"]
        == "Model-rephrased title"
    )
```

Extract the repeated construct-model fixture in the same test module:

```python
def _construct_model_update(
    *,
    source_id="src-valid",
    citation_label="Retrieved source title",
):
    return {
        "construct_model": {
            "theoretical_framework": "Retrieved framework",
            "framework_rationale": "Supported by retrieved evidence.",
            "source_references": [{
                "source_id": source_id,
                "framework": "Framework",
                "citation_label": citation_label,
                "supported_claim": "Facet structure",
            }],
            "dimensions": [{
                "source_facets": [{
                    "source_id": source_id,
                    "framework": "Framework",
                    "facet_code": None,
                    "facet_name": "Facet",
                    "mapping_type": "adapted",
                    "rationale": "Adapted for PSJT.",
                }],
            }],
        },
    }
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
python -m pytest -q tests\test_theory_search.py::test_construct_model_canonicalizes_citation_label_from_source_id
```

Expected: collection/import failure because `canonicalize_construct_model_sources` does not exist.

- [ ] **Step 3: Implement minimal source canonicalization**

In `sjt_system/theory_search.py`, add:

```python
from copy import deepcopy


def canonicalize_construct_model_sources(
    update: Mapping[str, Any],
    evidence: list[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate = deepcopy(dict(update))
    model = candidate.get("construct_model")
    if not isinstance(model, Mapping):
        raise ValueError(
            "build_construct_model 必须返回对象类型的 construct_model"
        )
    evidence_by_id = {
        item.get("source_id"): item
        for item in evidence
        if isinstance(item, Mapping)
        and isinstance(item.get("source_id"), str)
    }
    references = model.get("source_references")
    if not isinstance(references, list) or not references:
        raise ValueError(
            "construct_model.source_references 必须是非空列表"
        )
    for reference in references:
        if not isinstance(reference, dict):
            raise ValueError("每条 source_reference 必须是对象")
        source_id = reference.get("source_id")
        source = evidence_by_id.get(source_id)
        if source is None:
            raise ValueError(f"构念模型引用了未检索来源：{source_id!r}")
        title = source.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"检索来源缺少有效标题：{source_id!r}")
        reference["citation_label"] = title
    return candidate
```

- [ ] **Step 4: Run the targeted theory tests and verify GREEN**

Run:

```powershell
python -m pytest -q tests\test_theory_search.py
```

Expected: all tests pass.

---

### Task 2: Repair invalid construct-model source IDs

**Files:**
- Modify: `sjt_system/execute.py`
- Modify: `sjt_system/graph.py`
- Create: `tests/test_construct_model_repair.py`

- [ ] **Step 1: Write a failing test for one invalid source followed by a valid source**

Create `tests/test_construct_model_repair.py` with a fake construct agent that
returns an invalid `source_id` on its first call and a valid one on its second.
The key assertion is:

```python
result = asyncio.run(
    execute_module.execute_construct_model_with_repair(route, state)
)

assert len(agent.inputs) == 2
assert "src-valid" in agent.inputs[1]["input_data"]["validation_feedback"]
assert (
    result["state_update"]["construct_model"]["source_references"][0][
        "citation_label"
    ]
    == "Retrieved source title"
)
assert result["repair_attempt_count"] == 1
```

The test state must contain:

```python
state["theory_evidence"] = [{
    "source_id": "src-valid",
    "title": "Retrieved source title",
}]
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
python -m pytest -q tests\test_construct_model_repair.py
```

Expected: failure because `execute_construct_model_with_repair` does not exist.

- [ ] **Step 3: Implement bounded construct-model repair**

In `sjt_system/execute.py`, add a dedicated function that:

1. invokes `construct_agent`;
2. validates that `state_update` is a dictionary;
3. canonicalizes labels with `canonicalize_construct_model_sources`;
4. runs `validate_construct_model_sources`;
5. on `ValueError`, supplies `validation_feedback`,
   `previous_invalid_candidate`, and `valid_source_ids` for the next attempt;
6. stops after `MAX_ITEM_OUTPUT_REPAIR_ATTEMPTS + 1` total attempts.

The loop shape must be:

```python
async def execute_construct_model_with_repair(
    route: PSJTRouteDecision,
    state: PSJTState,
) -> dict[str, Any]:
    context = {
        "state": state,
        "target_item_id": route.get("target_item_id"),
        "target_blueprint_cell_id": route.get(
            "target_blueprint_cell_id"
        ),
    }
    input_data = dict(context)
    last_error: ValueError | None = None
    for repair_attempt in range(MAX_ITEM_OUTPUT_REPAIR_ATTEMPTS + 1):
        result = await _ainvoke_model(
            construct_agent,
            {"input_data": input_data},
            job_label="build_construct_model",
        )
        try:
            proposed_update = result.get("state_update")
            if not isinstance(proposed_update, dict):
                raise ValueError(
                    "Agent 输出缺少有效的 state_update"
                )
            canonical = canonicalize_construct_model_sources(
                proposed_update,
                state.get("theory_evidence") or [],
            )
            validate_construct_model_sources(
                canonical,
                state.get("theory_evidence") or [],
            )
            return {
                **result,
                "state_update": canonical,
                "repair_attempt_count": repair_attempt,
            }
        except ValueError as exc:
            last_error = exc
            if repair_attempt >= MAX_ITEM_OUTPUT_REPAIR_ATTEMPTS:
                break
            valid_source_ids = [
                source.get("source_id")
                for source in state.get("theory_evidence") or []
                if isinstance(source, dict)
            ]
            emit_progress({
                "type": "output_repair",
                "job_label": "build_construct_model",
                "attempt": repair_attempt + 2,
                "max_attempts": MAX_ITEM_OUTPUT_REPAIR_ATTEMPTS + 1,
                "reason": str(exc),
            })
            input_data = {
                **context,
                "validation_feedback": str(exc),
                "valid_source_ids": valid_source_ids,
                "previous_invalid_candidate": result,
            }
    raise ValueError(
        "构念建模结构输出自动修复"
        f"{MAX_ITEM_OUTPUT_REPAIR_ATTEMPTS}次后仍未通过校验："
        f"{last_error}"
    )
```

Route `build_construct_model` through this function in `execute_agent`.
Keep the graph-level source validation as defense in depth.

- [ ] **Step 4: Add and run a repair-exhaustion test**

The fake agent always returns `src-invalid`. Assert four calls and a final
`ValueError` mentioning automatic repair exhaustion.

Run:

```powershell
python -m pytest -q tests\test_construct_model_repair.py tests\test_theory_search.py
```

Expected: all tests pass.

---

### Task 3: Add bounded timeout retry for ordinary model calls

**Files:**
- Modify: `sjt_system/agent/API.py`
- Create: `sjt_system/model_retry.py`
- Modify: `sjt_system/execute.py`
- Modify: `sjt_system/graph.py`
- Modify: `tests/test_structured_output.py`
- Create: `tests/test_model_retry.py`

- [ ] **Step 1: Write failing configuration tests**

Add:

```python
def test_model_request_max_attempts_defaults_to_two(monkeypatch):
    monkeypatch.delenv("MODEL_REQUEST_MAX_ATTEMPTS", raising=False)
    assert get_model_request_max_attempts() == 2


def test_model_request_max_attempts_is_configurable(monkeypatch):
    monkeypatch.setenv("MODEL_REQUEST_MAX_ATTEMPTS", "3")
    assert get_model_request_max_attempts() == 3


@pytest.mark.parametrize("value", ["0", "6", "1.5", "invalid"])
def test_model_request_max_attempts_rejects_invalid_values(
    monkeypatch, value
):
    monkeypatch.setenv("MODEL_REQUEST_MAX_ATTEMPTS", value)
    with pytest.raises(ValueError):
        get_model_request_max_attempts()
```

- [ ] **Step 2: Run configuration tests and verify RED**

Expected: import failure because the getter does not exist.

- [ ] **Step 3: Implement the configuration getter**

In `sjt_system/agent/API.py` add:

```python
DEFAULT_MODEL_REQUEST_MAX_ATTEMPTS = 2


def get_model_request_max_attempts() -> int:
    raw_value = os.getenv(
        "MODEL_REQUEST_MAX_ATTEMPTS",
        str(DEFAULT_MODEL_REQUEST_MAX_ATTEMPTS),
    )
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "MODEL_REQUEST_MAX_ATTEMPTS must be an integer"
        ) from exc
    if value < 1 or value > 5:
        raise ValueError(
            "MODEL_REQUEST_MAX_ATTEMPTS must be between 1 and 5"
        )
    return value
```

- [ ] **Step 4: Write failing timeout-retry behavior tests**

In `tests/test_model_retry.py`, use a real fake Runnable whose `ainvoke`
raises `TimeoutError` once and then returns:

```python
result = asyncio.run(
    ainvoke_model_with_retry(
        agent,
        {"input_data": {}},
        job_label="test job",
        timeout_seconds=1,
        max_attempts=2,
        backoff_seconds=0,
    )
)
assert result == {"ok": True}
assert agent.calls == 2
```

Add separate tests for:

- two timeouts raise after two calls;
- `ValueError` is not retried;
- a `request_retry` event contains `attempt=2` and `max_attempts=2`.

- [ ] **Step 5: Run retry tests and verify RED**

Expected: module import failure because `sjt_system.model_retry` does not exist.

- [ ] **Step 6: Implement the retry helper**

Create `sjt_system/model_retry.py` with:

```python
async def ainvoke_model_with_retry(
    agent: Runnable,
    input_data: dict[str, Any],
    *,
    job_label: str,
    timeout_seconds: float | None = None,
    max_attempts: int | None = None,
    backoff_seconds: float = 1.0,
) -> Any:
    timeout_seconds = (
        get_model_request_timeout_seconds()
        if timeout_seconds is None
        else timeout_seconds
    )
    max_attempts = (
        get_model_request_max_attempts()
        if max_attempts is None
        else max_attempts
    )
    for attempt in range(1, max_attempts + 1):
        try:
            return await asyncio.wait_for(
                agent.ainvoke(input_data),
                timeout=timeout_seconds,
            )
        except TimeoutError as exc:
            if attempt >= max_attempts:
                emit_progress({
                    "type": "request_timeout",
                    "job_label": job_label,
                    "timeout_seconds": timeout_seconds,
                })
                raise TimeoutError(
                    f"{job_label} 请求超过 {timeout_seconds:g} 秒，"
                    f"已尝试 {max_attempts} 次"
                ) from exc
            emit_progress({
                "type": "request_retry",
                "job_label": job_label,
                "attempt": attempt + 1,
                "max_attempts": max_attempts,
                "reason": (
                    f"单次请求超过 {timeout_seconds:g} 秒"
                ),
            })
            if backoff_seconds:
                await asyncio.sleep(backoff_seconds)
```

- [ ] **Step 7: Integrate without duplicating retry logic**

Make `execute._ainvoke_model` delegate to
`ainvoke_model_with_retry`. Replace the Router's synchronous
`asyncio.to_thread(router_agent.invoke, ...)` call with:

```python
raw_decision = await ainvoke_model_with_retry(
    router_agent,
    {"input_data": state},
    job_label="工作流路由",
)
```

This avoids leaving an uncancellable background thread running after timeout.

- [ ] **Step 8: Run targeted retry and workflow tests**

Run:

```powershell
python -m pytest -q tests\test_model_retry.py tests\test_structured_output.py tests\test_review_aggregation.py tests\test_theory_search.py
```

Expected: all tests pass.

---

### Task 4: Persist and normalize run checkpoints

**Files:**
- Create: `sjt_system/run_checkpoint.py`
- Create: `tests/test_run_checkpoint.py`

- [ ] **Step 1: Write failing checkpoint round-trip tests**

Cover:

```python
path = save_run_checkpoint(state, checkpoint_root=tmp_path)
loaded = load_run_checkpoint(path)
assert loaded["state"] == state
assert loaded["schema_version"] == 1
```

Also assert:

- the final target is `<tmp_path>/<run_id>.json`;
- no `<run_id>.json.tmp` remains;
- `find_latest_resumable_checkpoint` ignores `completed` and `stopped`;
- corrupted JSON and mismatched `run_id` raise `ValueError`;
- unsupported schema versions raise `ValueError`.

- [ ] **Step 2: Run checkpoint tests and verify RED**

Expected: import failure because `sjt_system.run_checkpoint` does not exist.

- [ ] **Step 3: Implement atomic checkpoint persistence**

Create constants and functions:

```python
CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_CHECKPOINT_ROOT = Path("outputs/run_checkpoints")
TERMINAL_STATUSES = {"completed", "stopped"}


def save_run_checkpoint(
    state: Mapping[str, Any],
    *,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> Path:
    run_id = state.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("运行状态缺少有效 run_id")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    target = checkpoint_root / f"{run_id}.json"
    temporary = checkpoint_root / f"{run_id}.json.tmp"
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "saved_at": utc_timestamp(),
        "is_terminal": state.get("status") in TERMINAL_STATUSES,
        "state": dict(state),
    }
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
```

`load_run_checkpoint` must validate every envelope invariant before returning.
`find_latest_resumable_checkpoint` must sort valid nonterminal files by
`saved_at`, descending.

- [ ] **Step 4: Write and verify a failing resume-normalization test**

Start from a failed state containing an item pool, errors, history, and all
pending fields. Assert:

```python
resumed = prepare_resumed_state(state)
assert resumed["status"] == "running"
assert resumed["route"] is None
assert resumed["item_pool"] == state["item_pool"]
assert resumed["errors"] == state["errors"]
assert resumed["execution_history"] == state["execution_history"]
assert resumed["pending_state_update"] is None
assert state["status"] == "failed"
```

- [ ] **Step 5: Implement minimal resume normalization**

Use `deepcopy`, set `status` to `running`, set `route` and every pending or
user-decision field listed in the design to `None`, and preserve all committed
business fields and audit history.

- [ ] **Step 6: Run checkpoint tests and verify GREEN**

Run:

```powershell
python -m pytest -q tests\test_run_checkpoint.py
```

Expected: all tests pass.

---

### Task 5: Integrate checkpoint saving and CLI resume

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app_output.py`

- [ ] **Step 1: Write a failing test that saves every merged graph update**

Use a fake graph yielding two updates and a temporary checkpoint root:

```python
result = asyncio.run(
    app_module.run_with_trace(
        initial_state,
        checkpoint_root=tmp_path,
    )
)
saved = load_run_checkpoint(tmp_path / "run-save.json")
assert saved["state"] == result
```

- [ ] **Step 2: Run the test and verify RED**

Expected: `run_with_trace` does not accept `checkpoint_root`.

- [ ] **Step 3: Add checkpoint persistence to `run_with_trace`**

Add an optional `checkpoint_root` parameter. Immediately after each
`result.update(update)`, call:

```python
if checkpoint_root is not None:
    save_run_checkpoint(
        result,
        checkpoint_root=Path(checkpoint_root),
    )
```

Passing `None` keeps unit tests and library callers side-effect free. `main`
must pass `DEFAULT_CHECKPOINT_ROOT`.

- [ ] **Step 4: Write failing CLI selection tests**

Test two branches of a new `select_start_state` helper:

- input `"1"` returns `prepare_resumed_state(latest["state"])`;
- input `"2"` invokes the new-state factory;
- no checkpoint invokes the new-state factory without prompting.

The resume prompt output must include current phase, item count, and latest
error without dumping the full State.

- [ ] **Step 5: Implement startup resume selection**

Add:

```python
def select_start_state(
    new_state_factory: Callable[[], dict],
    *,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict:
    latest = find_latest_resumable_checkpoint(checkpoint_root)
    if latest is None:
        return new_state_factory()
    # Print concise metadata and accept only 1 or 2.
    # 1 resumes; 2 creates a new run.
```

Refactor the hard-coded `create_initial_state` call in `main` into the factory,
then pass the selected state to `run_with_trace`.

- [ ] **Step 6: Add an integration regression test for no duplicate accepted items**

Construct a failed checkpoint whose `item_pool` contains `item-1`, restore it,
and run a fake graph update that proceeds to audit. Assert the persisted
`item_pool` still contains exactly one `item-1`.

- [ ] **Step 7: Run app and checkpoint tests**

Run:

```powershell
python -m pytest -q tests\test_app_output.py tests\test_run_checkpoint.py
```

Expected: all tests pass with no files written outside `tmp_path`.

---

### Task 6: Full verification and handoff

**Files:**
- Verify all modified files
- Update: `README.md` only if runtime configuration is undocumented

- [ ] **Step 1: Document new runtime controls and recovery behavior**

Add concise README entries for:

- `MODEL_REQUEST_MAX_ATTEMPTS`;
- `MODEL_REQUEST_TIMEOUT_SECONDS`;
- checkpoint location;
- startup resume choice;
- the fact that pre-feature failed runs cannot be recovered.

- [ ] **Step 2: Run syntax verification**

Run:

```powershell
python -m py_compile app.py sjt_system\agent\API.py sjt_system\execute.py sjt_system\graph.py sjt_system\model_retry.py sjt_system\run_checkpoint.py sjt_system\theory_search.py
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Run the complete test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Inspect the final diff**

Run:

```powershell
git diff -- app.py README.md sjt_system tests docs
```

If Git metadata remains unavailable, inspect the exact modified files and
report that version-control diff/commit verification could not be performed.

- [ ] **Step 5: Report behavioral limitations**

State explicitly:

- only timeouts are automatically retried;
- invalid source IDs use bounded structured-output repair;
- checkpoints recover committed step-level state, not an in-flight HTTP call;
- old runs created before this change are not recoverable.
