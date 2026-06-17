# WING Readonly Runner Verifier Summary Surface Gate

## Context

The WING read-only preflight runner validates the detached no-WING artifact
manifest verifier before any backend/WING WebSocket opens. The verifier report
also mirrors the handoff-plan verifier summary surface, including its mismatch
list, match flag, and authority-denial flags.

## Options considered

1. Keep only generic verifier report value mismatch diagnostics.
2. Add dedicated violations for handoff verifier summary-surface mismatches.

## Decision

Add dedicated pre-WebSocket violations for these verifier summary-surface
fields:

- `wing_readonly_handoff_plan_verifier_summary_surface_mismatches`
- `wing_readonly_handoff_plan_verifier_summary_surface_match`
- `wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed`
- `wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted`

## Why this won

The handoff summary surface is the operator-facing bridge between no-WING
evidence and the later real preflight. If it is stale or authority-widening, the
runner should surface that root cause before transport opens.

## Rejected alternatives

Relying on generic report value mismatch was rejected because it does not
distinguish handoff summary-surface drift from ordinary artifact consistency
defects.

## Implementation plan

- Extend `tools/wing_readonly_preflight_runner.py` with dedicated summary
  surface mismatch violations.
- Add a pre-WebSocket regression test that mocks stale and authority-widening
  summary-surface fields.

## Test plan

- Focused runner verifier tests.
- Focused WING/no-WING tests.
- Standard full pytest command.

## Risks and rollback

Risk is limited to stricter offline/pre-WebSocket diagnostics. Rollback is to
remove the dedicated violations while preserving the generic verifier failure.
