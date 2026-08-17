"""Prompt for minimal IPIP-backed behavior evidence."""


BEHAVIOR_EVIDENCE_PROMPT = """
You extract the distinct behavior-evidence dimensions supported for one
personality facet.

Your task is to transform the supplied facet profile and IPIP source items into
a small set of construct-pure, observable, and behaviorally meaningful evidence
axes. These axes will later support weak-situation SJT development, but you must
not generate SJT content in this task.

Do not target a fixed number of records. Prefer fewer well-supported and
construct-pure dimensions over broader but contaminated coverage. Do not split
one behavioral mechanism into multiple records merely to increase the count.

INPUT AUTHORITY

Use the supplied facet profile to determine:

* the substantive meaning of the target facet;
* the intended high and low directions;
* the facet's hard boundaries and exclusions.

Use the supplied IPIP items to determine:

* which behavioral content is actually supported;
* which observable processes may be claimed;
* which source_item_ids may be cited.

A behavioral axis may be returned only when both conditions are satisfied:

1. it is consistent with the supplied facet profile; and
2. it is directly supported by the wording or immediate behavioral implication
   of the cited IPIP items.

Do not first invent a theoretically plausible mechanism from the facet profile
and then attach loosely related source_item_ids. The facet profile defines the
construct boundary, but the source items define the admissible evidence.

TASK SCOPE

Extract behavior-evidence dimensions only.

Do not generate:

* situations or scenarios;
* actors or roles;
* item stems;
* response options;
* scores or scoring keys;
* reviews or quality ratings;
* rationales or explanations;
* omitted-item reports;
* workflow metadata;
* behavior IDs.

Behavior IDs, if required, will be assigned deterministically by downstream
code.

OUTPUT STRUCTURE

Return one structured object containing a behavior_evidence array.

Each evidence record must contain exactly these fields:

* behavior_dimension;
* observable_behavior;
* high_expression;
* low_expression;
* boundary_condition;
* source_item_ids.

Use the following structure:

{{
"behavior_evidence": [
{{
"behavior_dimension": "...",
"observable_behavior": "...",
"high_expression": "...",
"low_expression": "...",
"boundary_condition": "...",
"source_item_ids": ["..."]
}}
]
}}

Do not add any other fields.

BEHAVIORAL AXIS REQUIREMENTS

Each behavior_dimension must:

* express one coherent behavioral continuum;
* primarily distinguish the target facet;
* use the same psychological mechanism at both ends;
* remain at a consistent level of abstraction;
* be concise but substantively specific;
* avoid evaluative labels such as good, bad, healthy, unhealthy, moral,
  immoral, appropriate, inappropriate, mature, or problematic.

The two endpoints must represent credible differences on the same axis. Do not
pair behaviors that differ simultaneously in morality, ability, effort,
emotional stability, interpersonal warmth, or another construct unless that
difference is itself central to the target facet.

Do not combine several loosely related behaviors into one record merely because
they have the same evaluative direction. If the behaviors reflect different
mechanisms, separate them only when each mechanism has sufficient direct source
support. If they reflect the same mechanism in different settings, merge them.

CONSTRUCT PURITY

Each axis must be explained more strongly by the target facet than by:

* another personality facet;
* general intelligence or verbal ability;
* knowledge, expertise, or educational opportunity;
* physical capacity or health;
* compliance with social expectations;
* general morality or social desirability;
* temporary mood or situational pressure;
* role requirements or institutional rules.

Do not use a broad socially valued outcome as the behavioral dimension when the
target facet concerns a more specific psychological process.

For example, do not automatically convert source-item references to mistakes,
rule violations, conflict, helping, achievement, emotional expression, or social
participation into broad axes such as:

* responsible versus irresponsible;
* prosocial versus antisocial;
* competent versus incompetent;
* lawful versus unlawful;
* cooperative versus harmful;
* confident versus insecure.

Retain such content only when the observable distinction is genuinely
constitutive of the target facet. When the target facet itself concerns
dutifulness, achievement, self-discipline, competence, order, altruism,
compliance, or another socially valued domain, preserve the facet-relevant
behavior but remove moralized, exaggerated, or globally evaluative wording.

Do not eliminate valid facet content merely because one endpoint is generally
more socially desirable. Preserve the construct's true direction, but express
both endpoints as plausible behavioral tendencies rather than an ideal person
versus a deficient person.

EVIDENCE ADMISSIBILITY

Every cited source_item_id must directly support the same behavioral mechanism
described in the record.

Do not cite an item merely because:

* it belongs to the target facet;
* it shares the same positive or negative scoring direction;
* it is thematically related;
* it helps increase source coverage;
* its surface example can be reinterpreted into a preferred mechanism.

Do not invent an information, communication, emotional, motivational, or
interpersonal process to make a contaminated source item appear
construct-pure. If the required process is not present or directly implied in
the source item, omit that item.

It is acceptable for some supplied source items not to appear in the final
output.

A record may be supported by a single source item when that item expresses a
distinct, unambiguous, and construct-relevant mechanism. Prefer converging
support from multiple items when available, but do not merge semantically
different items merely to obtain multiple sources.

Do not reuse one source item across multiple records unless the item clearly
contains two separable behavioral processes. When uncertain, assign the item
to one primary axis only.

OBSERVABILITY

observable_behavior must describe what a person observably does, says,
communicates, selects, avoids, repeats, changes, withholds, or reports.

Observable evidence may include:

* overt actions;
* verbal or written communication;
* repeated choices;
* response patterns across comparable situations;
* self-reported thoughts or feelings when the person explicitly communicates
  them.

Do not treat an inaccessible private motive as an observable behavior.

Do not directly infer:

* sincerity or insincerity;
* genuine or false concern;
* hidden motives;
* malicious intent;
* unconscious goals;
* internal moral character.

When a latent process involves motives or internal states, define it through
observable indicators, such as:

* consistency between words and actions;
* differences across audiences;
* timing of behavior relative to a request or reward;
* repeated changes in stated reasons;
* information explicitly included or omitted;
* persistence or withdrawal across comparable conditions;
* statements the person makes about their own experience.

Do not use a single ambiguous act as definitive evidence when several
alternative explanations are equally plausible.

ENDPOINT CONSTRUCTION

high_expression and low_expression must:

* represent the supplied facet's correct high and low directions;
* describe the same type of behavior in comparable conditions;
* be legal, feasible, and psychologically credible;
* be possible in ordinary, non-extreme situations;
* avoid cartoonish, malicious, reckless, heroic, or saint-like conduct;
* avoid explicit diagnostic or moral labels;
* avoid directly stating the underlying trait;
* be independently phrased rather than simple grammatical negations.

Both endpoints should be usable as latent anchors for later weak-situation SJT
design.

Do not make the high-expression endpoint:

* universally optimal;
* free of costs or trade-offs;
* maximally competent in addition to expressing the target trait;
* unusually self-sacrificing;
* perfectly regulated or flawless.

Do not make the low-expression endpoint:

* obviously illegal or unethical;
* intentionally harmful without source support;
* irrational or incompetent;
* globally irresponsible;
* socially absurd;
* dependent on an implausibly explicit admission of bad motives.

Where the construct and evidence support it, preserve realistic functional
tension between the endpoints. Different trait levels may involve different
preferences, priorities, sensitivities, thresholds, communication styles, or
risk tolerances.

However, do not invent benefits, costs, or compromises solely to make the
endpoints appear equally attractive.

Avoid replacing obvious misconduct with a legal loophole, technical
compliance, or vague gray-area behavior when the underlying distinction still
primarily measures rule compliance, morality, or another construct rather than
the target facet.

Do not assume that the following are inherently high or low expression without
facet-specific evidence:

* direct versus indirect communication;
* fast versus slow action;
* emotional versus unemotional expression;
* social participation versus solitude;
* persistence versus stopping;
* adherence versus flexibility;
* confidence versus hesitation;
* novelty versus familiarity;
* competition versus cooperation;
* self-disclosure versus privacy.

Their meaning depends on the target facet, source evidence, context, and the
specific psychological mechanism involved.

BOUNDARY CONDITIONS

boundary_condition must identify concrete circumstances in which the described
behavior should not support an inference about the target facet.

Consider relevant alternative explanations such as:

* objective danger or unusually high stakes;
* temporary illness, fatigue, grief, distress, or acute stress;
* lack of knowledge, skill, authority, access, or opportunity;
* cultural communication norms;
* formal role requirements;
* legal, ethical, safety, or confidentiality obligations;
* coercion or power imbalance;
* realistic resource constraints;
* a clearly unreasonable, harmful, or invalid demand;
* behavior driven more strongly by another personality facet.

Do not use generic boilerplate such as “context may matter.” State the actual
condition and why it weakens the inference.

A boundary condition should not excuse every low-expression behavior. It should
identify a plausible alternative explanation that specifically applies to the
axis.

CLUSTERING AND DISTINCTNESS

Before returning the output:

1. Group source items by their underlying behavioral mechanism, not merely by
   similar wording or scoring direction.

2. Merge records that differ only by setting, actor, object, or superficial
   phrasing.

3. Keep records separate only when they involve meaningfully different
   observable processes.

4. Reject records that are primarily synonyms of another record.

5. Reject records whose distinction depends more strongly on another facet,
   ability, morality, social convention, or situational demand.

6. Reject records whose endpoints cannot both be expressed as plausible,
   observable behaviors on the same continuum.

7. Reject records that require unsupported motive inference.

8. Reject records whose source items do not directly support the proposed
   mechanism.

9. Do not attempt to cover every source item.

10. Do not target a predetermined number of records.

FINAL INTERNAL CHECK

Silently verify every proposed record against all of the following questions:

* Is this one coherent behavioral axis?
* Does the target facet explain the distinction better than another construct?
* Is every cited source item direct evidence for this specific mechanism?
* Are the two endpoints on the same continuum and at the same level of
  abstraction?
* Are both endpoints observable or communicable?
* Have inaccessible motives been replaced with behavioral indicators?
* Are both endpoints plausible in ordinary situations?
* Is the low endpoint free from obvious wrongdoing or caricature unless such
  content is indispensable to the target facet and directly supported?
* Is the high endpoint free from idealized perfection or unrelated competence?
* Does the boundary condition identify a specific competing explanation?
* Is this record distinct from all other returned records?
* Would this axis remain meaningful if evaluative words were removed?

If any answer is no, revise, merge, or discard the record.

Return only the structured object. Do not include markdown, commentary,
explanations, headings, or text outside the object.
""".strip()
