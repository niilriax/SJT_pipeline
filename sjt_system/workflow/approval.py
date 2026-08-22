"""Normalize user decisions and commit approved state updates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sjt_system.state import UserDecision


_VALID_DECISIONS = {
    "approve",
    "edit",
    "regenerate",
    "answer",
    "accept_suggestions",
    "confirm",
    "revise",
    "stop",
}
_PROTECTED_FIELDS = {
    "errors",
    "execution_history",
    "max_steps",
    "pending_action",
    "pending_state_changes",
    "pending_state_update",
    "pending_summary",
    "route",
    "run_id",
    "step_count",
    "user_decision",
    "user_feedback",
}


def normalize_user_decision(
    raw_decision: Any,
    pending_update: Mapping[str, Any],
) -> UserDecision:
    """校验恢复命令，防止用户补丁覆盖工作流内部字段。"""

    if not isinstance(raw_decision, Mapping):
        raise ValueError("用户决策必须是一个对象")

    decision = raw_decision.get("decision")
    if decision not in _VALID_DECISIONS:
        raise ValueError("不支持的用户决策")

    feedback = raw_decision.get("feedback")
    if feedback is not None and not isinstance(feedback, str):
        raise ValueError("feedback 必须是字符串或 null")
    if decision in {"answer", "revise"} and not (feedback or "").strip():
        raise ValueError(f"{decision} 决策必须提供自然语言内容")

    state_patch = raw_decision.get("state_patch")
    if decision == "edit":
        if not isinstance(state_patch, Mapping) or not state_patch:
            raise ValueError("edit 决策必须提供非空 state_patch")

        patch_fields = set(state_patch)
        protected = patch_fields & _PROTECTED_FIELDS
        if protected:
            raise ValueError(
                "不能修改工作流内部字段：" + ", ".join(sorted(protected))
            )

        unexpected = patch_fields - set(pending_update)
        if unexpected:
            raise ValueError(
                "只能修改本次 Agent 提议的字段："
                + ", ".join(sorted(unexpected))
            )
        normalized_patch: dict[str, Any] | None = dict(state_patch)
    else:
        normalized_patch = None

    return {
        "decision": decision,
        "feedback": feedback,
        "state_patch": normalized_patch,
    }


def build_committed_update(
    pending_update: Mapping[str, Any],
    decision: UserDecision,
) -> dict[str, Any]:
    """将用户允许的编辑合并进候选更新。"""

    committed = dict(pending_update)
    if decision["decision"] == "edit":
        committed.update(decision.get("state_patch") or {})
    return committed
