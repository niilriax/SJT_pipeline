# Balanced item-review design

## Decision

Replace the single comprehensive item-review Agent with two independent
specialists:

1. a construct reviewer;
2. a content and option-quality reviewer.

Cross-item similarity remains primarily deterministic. Similarity evidence is
provided to the content reviewer only; no permanent third similarity Agent is
introduced.

## Review flow

For each committed item version:

1. build separate construct-review and content-review inputs;
2. run both specialist Agents independently and concurrently;
3. compute deterministic cross-item findings;
4. aggregate the three result sources into the existing `PASS`, `REVISE`, or
   `REJECT` workflow decision;
5. preserve both expert reports in State and item history;
6. send `REVISE` items to local revision and `REJECT` items to regeneration.

Only an aggregate `PASS` may enter the candidate item pool. Every revised or
regenerated version receives a complete new review.

## Specialist boundaries

The construct reviewer receives the item, assigned construct dimension,
behavioral anchors, blueprint cell, and item specification. It reviews
construct alignment, situation activation, option unidimensionality,
contamination, scoring logic, and compliance with the assigned context slot.

The content reviewer receives the item, target-population and response
specification, blueprint constraints, deterministic quality findings, and the
most similar accepted items. It reviews clarity, realism, option quality,
social desirability, answer cues, language, fairness, and substantive
duplication.

Neither specialist receives the other specialist's output.

## Expert output and aggregation

Each specialist returns its reviewer type, decision, repair scope, findings,
and summary. Findings contain criterion, severity, evidence, finding, and
recommendation.

The program aggregates deterministically:

- both experts pass and no deterministic warning or blocking finding: `PASS`;
- either expert requests a local revision, or a deterministic warning exists:
  `REVISE`;
- either expert returns a validated rewrite-level rejection, or deterministic
  similarity identifies a near copy: `REJECT`.

An expert rejection is valid only when it includes a blocking finding and
`repair_scope="rewrite"`. A pass cannot contain a blocking finding.

## Approval behavior

This change preserves the current human-confirmation behavior. The combined
review result is presented as one pending review action, even though it
contains two independent expert reports.

## Testing

Tests cover independent inputs, concurrent specialist execution, deterministic
aggregation, invalid expert outputs, review-history preservation, and the
existing revise/regenerate routing.
