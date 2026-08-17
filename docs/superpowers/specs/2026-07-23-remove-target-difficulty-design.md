# Remove `target_difficulty` from the PSJT blueprint

## Decision

Remove the author-assigned `target_difficulty` field from the current
behavioral-tendency personality SJT workflow.

The current implementation defaults the field to `medium` and allows the
blueprint LLM to choose `low`, `medium`, or `high`, but it provides no
operational definition or empirical calibration. Keeping the field would imply
measurement precision that the workflow does not possess.

## Scope

Remove `target_difficulty` from:

- `BlueprintCell`;
- `ItemSpecification`;
- blueprint skeleton construction;
- item-specification expansion;
- blueprint prompts;
- blueprint candidate and full-blueprint validation;
- tests and fixtures that construct blueprint cells or item specifications.

Do not replace it with another author-assigned difficulty field.

## Preserved controls

The blueprint continues to control:

- construct-dimension coverage;
- planned generation and retention counts;
- context-category quotas;
- item-level context slots;
- scenario, option, and scoring constraints.

The executable blueprint is therefore defined by construct dimension, context
category, and planned item count rather than by an unsupported difficulty
label.

## Future empirical field

If pilot-response data later becomes available, item-level score distributions
or model-based parameters may be stored under explicitly empirical names. Such
fields must be derived from response data and must not reuse
`target_difficulty`.

## Compatibility and validation

This is an intentional schema change. Newly generated blueprints and items must
not contain `target_difficulty`. Existing in-memory or serialized blueprints
that contain the field may be read only if the loader ignores unknown fields;
the current repository has no persistent blueprint migration layer, so no
migration is included in this change.

All existing tests must pass after fixtures are updated, and a regression test
must assert that generated blueprint cells and item specifications omit the
field.
