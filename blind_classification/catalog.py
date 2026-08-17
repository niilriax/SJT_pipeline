"""Blind classification facet catalogs built from the versioned construct registry.

Two catalogs are supported:

- CATALOG_30 (default): all 30 NEO-PI-R facets from the registry. Chance
  baseline 1/30. Answers the full question "which facet is this item"?
- CATALOG_5 (legacy): the five facets of the Mussel gold set. Chance
  baseline 1/5. Kept for comparison with earlier runs.

Ground-truth label mapping:

    Mussel 开放性  -> openness_ideas                (O5)
    Mussel 责任心  -> conscientiousness_self_discipline (C5)
    Mussel 外倾性  -> extraversion_gregariousness   (E2)
    Mussel 宜人性  -> agreeableness_compliance      (A4)
    Mussel 神经质  -> neuroticism_self_consciousness (N4)
"""
from __future__ import annotations

from typing import Any

from sjt_system.authoring.construct_registry import resolve_construct_profile

ALL_FACET_IDS = [
    "agreeableness_altruism",
    "agreeableness_compliance",
    "agreeableness_modesty",
    "agreeableness_straightforwardness",
    "agreeableness_tender_mindedness",
    "agreeableness_trust",
    "conscientiousness_achievement_striving",
    "conscientiousness_competence",
    "conscientiousness_deliberation",
    "conscientiousness_dutifulness",
    "conscientiousness_order",
    "conscientiousness_self_discipline",
    "extraversion_activity",
    "extraversion_assertiveness",
    "extraversion_excitement_seeking",
    "extraversion_gregariousness",
    "extraversion_positive_emotions",
    "extraversion_warmth",
    "neuroticism_angry_hostility",
    "neuroticism_anxiety",
    "neuroticism_depression",
    "neuroticism_impulsiveness",
    "neuroticism_self_consciousness",
    "neuroticism_vulnerability",
    "openness_actions",
    "openness_aesthetics",
    "openness_fantasy",
    "openness_feelings",
    "openness_ideas",
    "openness_values",
]

# Legacy 5-class labels (short ids).
FACET_IDS_5 = [
    "openness_to_ideas",
    "self_discipline",
    "gregariousness",
    "compliance",
    "self_consciousness",
]

MUSSEL_FACET_KEYS = ["开放性", "责任心", "外倾性", "宜人性", "神经质"]

# Mussel facet key -> registry facet id (30-class ground truth).
MUSSEL_KEY_TO_REGISTRY_FACET = {
    "开放性": "openness_ideas",
    "责任心": "conscientiousness_self_discipline",
    "外倾性": "extraversion_gregariousness",
    "宜人性": "agreeableness_compliance",
    "神经质": "neuroticism_self_consciousness",
}

# Registry facet id -> 5-class short label (legacy mode only).
REGISTRY_TO_SHORT_FACET = {
    "extraversion_gregariousness": "gregariousness",
    "agreeableness_compliance": "compliance",
    "neuroticism_self_consciousness": "self_consciousness",
    "conscientiousness_self_discipline": "self_discipline",
    "openness_ideas": "openness_to_ideas",
}


def build_catalog(facet_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Resolve facet definitions from the registry for the given ids."""

    catalog: list[dict[str, Any]] = []
    for facet_id in facet_ids or ALL_FACET_IDS:
        profile = resolve_construct_profile(facet_id)
        facet = profile["facets"][0]
        catalog.append(
            {
                "facet_id": facet_id,
                "facet_name": facet.get("facet_name"),
                "facet_name_en": facet.get("facet_name_en"),
                "definition": facet.get("definition"),
                "high_behavior": facet.get("high_behavior"),
                "low_behavior": facet.get("low_behavior"),
                "common_confounds": facet.get("common_confounds") or [],
            }
        )
    return catalog


def catalog_text(catalog: list[dict[str, Any]]) -> str:
    lines = []
    for entry in catalog:
        lines.append(
            f"- {entry['facet_id']}（{entry['facet_name']}）："
            f"{entry['definition']}"
        )
        lines.append(f"    高特质表现：{entry['high_behavior']}")
        lines.append(f"    低特质表现：{entry['low_behavior']}")
        if entry["common_confounds"]:
            lines.append(f"    易混淆构念：{'、'.join(entry['common_confounds'])}")
    return "\n".join(lines)
