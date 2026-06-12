# WING Handoff Prestart Gate Final-Step Validation

## Context

The handoff packet operator sequence is ordered. The supervised prestart gate
must not be followed by a weaker proof instruction, otherwise reviewers could
see a valid gate while the final operator action is unsafe.

## Options considered

1. Keep same-step validation only.
2. Require the same-step supervised prestart gate to be the final operator
   sequence item.
3. Introduce a new packet schema field for the proof transition.

## Decision

Use option 2. The validator now fails if the accepted-candidate prestart gate
step is present but not final.

## Why this won

It aligns validation with the ordered operator flow without changing packet
schema.

## Rejected alternatives

- Same-step-only validation was rejected because it permits later conflicting
  instructions.
- Schema changes were deferred because this invariant can be enforced on the
  existing packet format.

## Implementation plan

- Track the gated step index.
- Fail validation when it is not the last operator sequence item.
- Render the index/finality fields in summary output.
- Add focused regression coverage.

## Test plan

- Focused handoff packet tests.
- Affected no-WING/WING tests.
- Full pytest suite.
- `git diff --check`.

## Risks and rollback

Risk is stricter validation for handoff packets with extra later instructions.
Rollback is safe by removing the final-step check, but that reopens ambiguous
operator guidance.
