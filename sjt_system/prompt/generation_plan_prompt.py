"""Prompt for one item skeleton selected by the two-way table."""


COMPACT_SKELETON_BATCH_PROMPT = """
Design one abstract PSJT item skeleton. A later agent writes the item.

Use only the supplied facet definition, behavior evidence, activation
mechanism, situation domain, actor relation, and event class. Preserve the
behavior evidence high/low direction, boundary condition, facet constraints,
and the supplied activation mechanism. Fit the target population.

Return exactly one item_skeleton with:
- situation_type: an abstract event type, not scenario prose;
- stakes_level: low, medium, or high;
- social_context: the relevant setting and role relationship;
- behavioral_tension: the decision tension that realizes the mechanism;
- option_structure: exactly low, medium_low, medium_high, and high, each with
  behavioral_tendency and psychological_function.
- Option design constraints: the 1-point behavior must be a plausible,
  rationalized or indirect form of avoidance, never a cartoon villain or
  blatant malicious act. Adjacent levels (1 vs 2, and 3 vs 4) must differ in
  visible action, timing, communication, or follow-through, not only in
  internal attitude or degree adverbs. The 4-point behavior must be
  principled and constructive with proportionate communication and boundaries,
  not blind self-sacrifice. All four options must be realistically feasible in
  the same information, resources, duties, and uncertainty.
- The situation must be a weak situation with trait-relevant cues. It must
  contain a real conflict that activates the target Behavior Evidence without
  explicit legal bans, severe moral violations, obvious punishment,
  authoritative commands, or extreme consequences that dictate one correct
  answer. Use at least one pair of partly reasonable demands (for example duty
  versus convenience, commitment versus temporary change, procedure versus
  cost, or another person's need versus limited resources). Do not collapse the item
  into right/wrong, legal/illegal, or responsible/malicious avoidance.

Do not return IDs, item text, response wording, scores, reviews, rationales, or
workflow fields. The four levels must face the same information, duties,
resources, costs, and uncertainty. Return only the requested structured object.
""".strip()
