# WING Handoff Prestart Gate Validator

## Context

The supervised proof plan requires accepted read-only preflight source ids to
have a valid accepted-candidate prestart trace. The handoff packet generator now
states that boundary, but the no-WING release validator must also reject stale
or manually edited packet artifacts that omit it.

## Options considered

1. Keep validating only operator sequence length.
2. Require supervised-proof transition markers in the operator sequence.
3. Add a new handoff packet schema field.

## Decision

Use option 2. Release validation now requires the handoff packet operator
sequence to mention `accepted_decision_ready`, `valid accepted-candidate
prestart trace`, and `WING_SUPERVISED_FADER_GAIN_PROOF`.

## Why this won

It is a small fail-closed validator change that protects the current packet
format from stale transition guidance.

## Rejected alternatives

- Length-only validation was rejected because it accepts stale wording.
- A new schema field was deferred because it would require broader artifact
schema churn for the same immediate safety check.

## Implementation plan

- Add required marker constants to `tools/no_wing_release_check.py`.
- Add validator detail and violation output.
- Add regression tests for valid and stale handoff packet operator sequences.

## Test plan

- Focused handoff packet tests.
- Affected WING/no-WING tests.
- Full pytest suite.
- `git diff --check`.

## Risks and rollback

Older handoff packet artifacts without the prestart trace marker will fail
release validation. Rollback is safe by removing the marker check and focused
test, but that would reopen the stale-artifact path.
