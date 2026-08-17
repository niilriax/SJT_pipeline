"""序贯贝叶斯问题发现：findings → 后验 → 动作。

核心（家族分组朴素贝叶斯）：
    posterior(p) ∝ prior(p) · ∏_family likelihood(该家族观察到的结果 | p)

每个 finding 家族对一道题只贡献一个结果：触发的成员，或"未触发"
（概率 = 1 − Σ该家族灵敏度）。互斥 finding 因此不会重复计数，
"未触发"本身也是证据（例如盲法分类正确会强力压低构念错位假说）。

决策流程（每次最多追问一步，形成序贯吸收证据的循环）：
1. 已知 findings → 后验；
2. 若 P_NONE 领跑且过阈值 → retain；
3. 若最高问题后验 ≥ THETA_ACT 且领先次名 ≥ MARGIN → 明确问题，
   按期望损失最小化在 RETAIN/REVISE_OPTIONS/REVISE_SCENARIO/REMOVE 中选动作；
4. 否则歧义 → 在未观察的候选指标家族里选期望信息增益最大者：
   增益 ≥ EPS_IG 则"先测该指标"，否则升级 INVESTIGATE（LLM 诊断）。
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from experiments.problem_discovery.taxonomy import (
    ACTIONS,
    CANDIDATE_FAMILIES,
    DEFAULT_SENSITIVITY,
    EPS_IG,
    FAMILIES,
    FAMILY_COST,
    FINDINGS,
    LOSS,
    MARGIN,
    PRIOR,
    SENSITIVITY,
    THETA_ACT,
)


def sensitivity(problem: str, finding_code: str) -> float:
    """P(finding 触发 | 问题存在)，缺省用 DEFAULT_SENSITIVITY。"""
    return SENSITIVITY[problem].get(finding_code, DEFAULT_SENSITIVITY)


def outcome_likelihood(outcome: str | None, family: str, problem: str) -> float:
    """该家族观察结果对某问题的似然。

    outcome 为触发的 finding code；None 表示整族未触发，
    概率 = 1 − Σ该族灵敏度（下限 0.05，避免数值零概率）。
    """
    if outcome is not None:
        return sensitivity(problem, outcome)
    fired_sum = sum(
        sensitivity(problem, code)
        for code, (fam, _) in FINDINGS.items()
        if fam == family
    )
    return max(0.05, 1.0 - fired_sum)


def posterior(
    fired: Iterable[str],
    observed_families: Iterable[str] = (),
) -> dict[str, float]:
    """给定已触发的 findings 与已观察的家族，返回问题后验分布。

    observed_families 必须是"确实算过该指标"的家族集合：
    没算过的家族既不按触发也不按未触发更新（缺失 ≠ 未触发）。
    """
    fired_codes = list(fired)
    fired_by_family: dict[str, str] = {}
    for code in fired_codes:
        if code not in FINDINGS:
            raise ValueError(f"未知 finding: {code}")
        fired_by_family.setdefault(FINDINGS[code][0], code)
    observed = set(observed_families)
    for family in fired_by_family:
        observed.add(family)
    unknown = observed - set(FAMILIES)
    if unknown:
        raise ValueError(f"未知 finding 家族: {sorted(unknown)}")

    weights: dict[str, float] = {}
    for problem in PRIOR:
        weight = math.log(PRIOR[problem])
        for family in FAMILIES:
            if family not in observed:
                continue
            outcome = fired_by_family.get(family)  # None = 未触发
            weight += math.log(outcome_likelihood(outcome, family, problem))
        weights[problem] = math.exp(weight)
    total = sum(weights.values())
    return {problem: weight / total for problem, weight in weights.items()}


def entropy(probs: dict[str, float]) -> float:
    return -sum(value * math.log2(value) for value in probs.values() if value > 0)


def posterior_given_outcome(
    prior: dict[str, float],
    family: str,
    outcome: str | None,
) -> dict[str, float]:
    weights = {
        problem: prob * outcome_likelihood(outcome, family, problem)
        for problem, prob in prior.items()
    }
    total = sum(weights.values())
    return {problem: weight / total for problem, weight in weights.items()}


def expected_information_gain(prior: dict[str, float], family: str) -> float:
    """IG(family) = H(后验) − E[H(后验 | 家族结果)]。

    医学诊断里"下一个该做的检查"就是这个量：谁最能区分
    当前还活着的竞争假说，就先测谁。
    """
    members = [code for code, (fam, _) in FINDINGS.items() if fam == family]
    outcomes: list[str | None] = [*members, None]
    outcome_probs: dict[str | None, float] = {}
    for outcome in outcomes:
        outcome_probs[outcome] = sum(
            prior[problem] * outcome_likelihood(outcome, family, problem)
            for problem in prior
        )
    expected_entropy = 0.0
    for outcome, prob in outcome_probs.items():
        if prob <= 0:
            continue
        expected_entropy += prob * entropy(
            posterior_given_outcome(prior, family, outcome)
        )
    return max(0.0, entropy(prior) - expected_entropy)


def expected_loss(action: str, probs: dict[str, float]) -> float:
    return sum(probs[problem] * LOSS[action][problem] for problem in probs)


def decide(
    fired: Iterable[str],
    observed_families: Iterable[str] = (),
) -> dict[str, Any]:
    """一次决策：返回后验、排名、决策带与动作（或下一步指标建议）。"""
    probs = posterior(fired, observed_families)
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    top_problem, top_value = ranked[0]
    second_value = ranked[1][1] if len(ranked) > 1 else 0.0
    observed = set(observed_families) | {
        FINDINGS[code][0] for code in fired if code in FINDINGS
    }

    if top_problem == "P_NONE" and top_value >= THETA_ACT:
        decision = "healthy"
        action = "RETAIN"
        suggestion = None
    elif top_value >= THETA_ACT and top_value - second_value >= MARGIN:
        decision = "act"
        action = min(ACTIONS, key=lambda a: expected_loss(a, probs))
        suggestion = None
    else:
        decision = "ambiguous"
        candidates = [
            family for family in CANDIDATE_FAMILIES if family not in observed
        ]
        gains = {family: expected_information_gain(probs, family) for family in candidates}
        scores = {
            family: gains[family] / FAMILY_COST[family] for family in candidates
        }
        if candidates and max(gains.values()) >= EPS_IG:
            suggestion = max(scores, key=scores.get)
            action = f"measure:{suggestion}"
        else:
            suggestion = None
            action = "INVESTIGATE"

    return {
        "posterior": probs,
        "ranked": ranked,
        "decision": decision,
        "action": action,
        "suggestion": suggestion,
        "information_gain": (
            {family: round(gains[family], 4) for family in gains}
            if decision == "ambiguous"
            else {}
        ),
    }
