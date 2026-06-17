# WING MVP Supervised Prestart Expected Artifact Trace Gate

## Context

The supervised WING MVP remains blocked on the first real read-only connection
proof. Before supervised fader/gain proof planning, the handoff packet already
requires a valid accepted-candidate prestart trace, artifact-root trace, and an
expected-artifact-bound accepted source id.

Before this change, the expected-artifact trace requirement was present in the
operator step and proof-plan summary fields, but the structured
`supervised_prestart_gate` did not carry a dedicated boolean requirement for it.

## Options considered

1. Keep expected-artifact trace as text-only in the final operator step.
2. Add a dedicated boolean to `supervised_prestart_gate`.
3. Create a separate expected-artifact gate object.

## Decision

Use option 2. The shared `expected_supervised_prestart_gate()` now requires
`requires_accepted_candidate_expected_artifact_trace: true`.

## Why this won

It keeps the unlock contract compact, machine-checkable, and aligned with the
existing prestart and artifact-root trace booleans. It does not change live
execution, backend startup, readiness closure, or WING transport behavior.

## Rejected alternatives

- Option 1 was rejected because operator text alone is weaker than a structured
  contract field.
- Option 3 was rejected because the existing supervised prestart gate is already
  the correct unlock boundary for fader/gain proof planning.

## Implementation plan

- Add the boolean to the shared expected supervised prestart gate.
- Require the marker in Current First Command baseline coverage.
- Mirror the marker in MVP/autoloop docs.
- Add focused regression assertions for no-WING artifacts, handoff packets, and
  Project Director surfaces.

## Test plan

- Focused contract/no-WING/director tests.
- no-WING quick check.
- Full pytest.
- Memory lint.
- `git diff --check`.

## Risks and rollback

Risk is limited to stricter offline handoff validation and updated fixtures.
Rollback is to remove the boolean, marker, and assertions. No live WING command
or mixer mutation is introduced.
