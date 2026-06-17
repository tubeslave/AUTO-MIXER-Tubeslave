# ADR: WING Read-Only Pre-WebSocket Readiness Denial

## Context

The first WING MVP blocker is still real read-only connection proof. Several
pre-WebSocket runner outputs already denied live authority and execution, but
not every offline or validation-only surface carried an explicit
`readiness_closure_allowed: false` field.

## Options considered

1. Rely on missing readiness fields plus `live_authority_granted: false`.
2. Add explicit readiness denial to pre-WebSocket offline and validation
   surfaces.
3. Delay the field until real hardware artifacts exist.

## Decision

Use option 2. Offline plans, blocked plan-only validation/prestart outputs, and
validation-only real-run summaries now carry `readiness_closure_allowed: false`.

## Why this won

The read-only proof chain must be easy to audit mechanically. Explicit
readiness denial prevents an offline plan, blocked validation output, or
validation-only success from being mistaken for readiness closure before a real
WING artifact and manual review exist.

## Rejected alternatives

- Option 1 leaves ambiguity for downstream report and gate consumers.
- Option 3 keeps the current gap in pre-hardware surfaces where the operator
  needs the clearest no-authority signal.

## Implementation plan

- Add `readiness_closure_allowed: false` to the offline preflight plan.
- Add the same field to plan-only blocked validation/prestart outputs.
- Add the same field to the base real-run summary and validation-only success
  output.
- Assert the field in representative plan-only, validation-only, and blocked
  validation tests.

## Test plan

- Targeted WING preflight runner tests for the changed surfaces.
- no-WING quick release check.
- Focused WING preflight, no-WING, handoff, and Project Director tests.
- `git diff --check`.
- Full pytest.

## Risks and rollback

The risk is limited to adding an explicit false safety field. Rollback is to
remove the field and its assertions, but that would reintroduce ambiguity in
pre-WebSocket readiness surfaces.
