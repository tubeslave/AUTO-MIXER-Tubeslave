# WING Handoff Prestart Gate Same-Step Validation

## Context

The WING handoff packet validator must prevent stale operator guidance before
physical supervised proof. Whole-sequence marker matching can pass when the
prestart phrase exists somewhere in the packet but not in the proof transition
step.

## Options considered

1. Keep whole-sequence marker matching.
2. Require all supervised prestart gate markers in one operator sequence step.
3. Require the gate markers only in the final operator sequence step.

## Decision

Use option 2. Validation now accepts a handoff packet only when one operator
sequence item contains all required supervised prestart gate markers.

## Why this won

It directly protects the operator transition sentence without depending on
weaker whole-packet text presence.

## Rejected alternatives

- Whole-sequence matching was rejected as too permissive.
- Final-step-only matching was deferred to avoid unnecessary ordering coupling
  while still enforcing the proof transition invariant.

## Implementation plan

- Compute and expose `operator_sequence_supervised_prestart_gate_step`.
- Fail validation when the step is missing.
- Add split-marker regression coverage.
- Add summary rendering for the accepted gate step.

## Test plan

- Focused handoff packet tests.
- Affected no-WING/WING tests.
- Full pytest suite.
- `git diff --check`.

## Risks and rollback

Risk is stricter validation for stale handoff packets. Rollback is safe by
returning to whole-sequence matching, but that reopens the misleading transition
case.
