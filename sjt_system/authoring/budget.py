"""Candidate-development accounting shared by reporting paths."""

from __future__ import annotations

from collections.abc import Mapping


def developed_candidate_ids(state: Mapping[str, Any]) -> set[str]:
    identifiers = {
        str(specification["specification_id"])
        for specification in state.get("item_specifications") or []
        if isinstance(specification, Mapping)
        and isinstance(specification.get("specification_id"), str)
        and specification["specification_id"]
    }
    identifiers.update(
        str(identifier)
        for identifier in state.get("blueprint_candidate_attempted_ids") or []
        if isinstance(identifier, str) and identifier
    )
    for field in (
        "item_pool",
        "frozen_item_bank",
        "removed_items",
        "rejected_items",
    ):
        identifiers.update(
            str(item["item_id"])
            for item in state.get(field) or []
            if isinstance(item, Mapping)
            and isinstance(item.get("item_id"), str)
            and item["item_id"]
        )
    return identifiers
