# Item-count-driven Situation Capacity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the requested final item count determine Expansion situation capacity and one-to-one Blueprint slots without forcing extra mechanisms.

**Architecture:** Pass a deterministic per-facet situation quota into Expansion, retry when the returned total is below that quota, and require Blueprint to bind exactly one unique situation reference per requested item. Retain existing IDs and compatibility fields while removing duplicated-slot fallback behavior.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, LangChain runnable agents.

---

### Task 1: Lock the Expansion capacity contract

**Files:**
- Modify: `tests/test_situation_space.py`
- Modify: `sjt_system/authoring/situation_space.py`

- [x] Add a failing test showing that `required_situation_count` reaches the Expansion agent and an under-capacity result is retried.
- [x] Run the focused test and confirm it fails because the parameter and capacity retry do not exist.
- [x] Add `required_situation_count` to `ensure_facet_expansion` and `_generate_expansion`, validate cached and newly generated capacity, and retry only under-capacity outputs.
- [x] Change mechanism situation lists to `min_length=1` without a fixed maximum and align the Prompt with the dynamic total quota.
- [x] Run the focused Expansion tests.

### Task 2: Enforce one Blueprint row per requested item

**Files:**
- Modify: `tests/test_generation_plan_v9.py`
- Modify: `sjt_system/authoring/generation_plan.py`
- Modify: `sjt_system/authoring/situation_space.py`

- [x] Add a failing test showing that insufficient unique references are rejected rather than converted into multiple slots for one row.
- [x] Run the focused test and confirm the old duplicated-slot behavior fails the new expectation.
- [x] Require `available_reference_count >= generation_total`, set `row_total = generation_total`, and set every cell's generation and retention count to 1.
- [x] Run generation-plan and situation-space focused tests.

### Task 3: Pass balanced quotas from the workflow

**Files:**
- Modify: `tests/test_blueprint_flow.py`
- Modify: `sjt_system/workflow/executor.py`

- [x] Add a failing test for deterministic balanced distribution of N required situations across selected facets.
- [x] Add the smallest quota helper and pass each quota to `ensure_facet_expansion`.
- [x] Run the focused workflow tests and compile modified modules.

### Task 4: Verify the MVP contract

**Files:**
- Test: `tests/test_situation_space.py`
- Test: `tests/test_generation_plan_v9.py`
- Test: `tests/test_blueprint_flow.py`

- [x] Run all local tests covering Expansion, Blueprint, and fixed blueprint progress.
- [x] Confirm the final diff changes no Behavior Evidence, Skeleton, Item Writer, review, simulation, or delivery code.
