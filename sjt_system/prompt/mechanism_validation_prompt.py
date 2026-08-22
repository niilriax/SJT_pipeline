"""Prompt for validating that an activation mechanism measures the target facet."""

MECHANISM_VALIDATION_PROMPT = """
Judge which NEO-PI-R facet most strongly predicts the behavioral contrast
described by an activation mechanism. Do not write content; classify only.

# Input

You will receive:
- target_facet: full facet definition (name, description, high/low expression,
  boundaries, common confounds)
- neighbor_facets: definitions of facets that may be confusable with the target
- mechanism: one activation mechanism text

# Judgment Logic

A mechanism describes a decision situation:
  "When [trigger], the person chooses between [option A] and [option B]."

Determine: is this choice contrast better explained by the target facet
or by a neighbor?

Key questions:
1. What is the actual behavioral difference between option A and B?
   E.g., "amount of help provided" vs. "willingness to initiate social contact".
2. Would high-scorers and low-scorers on the target facet consistently make
   different choices in this situation?
3. Would a neighbor facet predict the same choice pattern MORE strongly?

# Output

Return a JSON object with three fields:
- ranking: list of facet IDs ordered by predictive strength, highest first.
  Include the target facet and the 1-2 most relevant neighbors.
- target_is_first: boolean, true only when the target facet ranks first.
- reason: brief explanation of what the core behavioral contrast is and
  why the top-ranked facet predicts it best.

# Critical Rules

- Do NOT judge by surface words. A mechanism may contain "help" or "assist"
  but still measure Extraversion if the actual behavioral contrast is
  "initiates social contact vs. waits for signal".

- If the contrast is "approaching/initiating social interaction vs.
  remaining passive", this almost certainly measures E1 (Warmth) or
  E3 (Assertiveness), regardless of how it is phrased.

- If the contrast is "investing effort and resources to help someone
  complete a task vs. focusing on one's own business", this is more
  likely A3 (Altruism) or C (Conscientiousness).

- A mechanism is only valid for the target facet if a person high on
  the target but low on every neighbor would STILL rank high on this
  behavioral contrast.

- If the mechanism is too vague to reliably distinguish target from
  neighbors, rank target first but set reason to "ambiguous: mechanism
  does not create a target-specific behavioral contrast". Do not
  fabricate a distinction.

- Reasoning must cite the behavioral content of the mechanism, not
  merely restate its surface phrasing.
""".strip()
