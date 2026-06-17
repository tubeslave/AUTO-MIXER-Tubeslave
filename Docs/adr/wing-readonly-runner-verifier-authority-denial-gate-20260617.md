# WING Readonly Runner Verifier Authority Denial Gate

## Context

The WING read-only preflight runner validates the detached no-WING artifact
manifest verifier before any backend/WING WebSocket opens. The verifier report
must never grant readiness closure or live authority; those permissions require
real WING evidence and manual review later in the MVP flow.

## Options considered

1. Keep only generic verifier report value mismatch diagnostics.
2. Add dedicated violations when the verifier report claims readiness closure
   or live authority.

## Decision

Add dedicated pre-WebSocket violations for non-false verifier authority flags:

- `prestart_summary_artifact_manifest_verifier_readiness_closure_allowed_not_false`
- `prestart_summary_artifact_manifest_verifier_live_authority_granted_not_false`

## Why this won

Authority claims are safety-critical and should be visible without forcing the
operator to infer them from a generic value mismatch list. The runner already
fails closed; the new violations make the authority-denial reason explicit.

## Rejected alternatives

Relying on generic report value mismatch was rejected because it hides a live
authority boundary violation among ordinary verifier report mismatches.

## Implementation plan

- Extend `tools/wing_readonly_preflight_runner.py` with dedicated authority
  denial violations.
- Add a pre-WebSocket regression test that mocks verifier readiness and live
  authority claims.

## Test plan

- Focused runner verifier tests.
- Focused WING/no-WING tests.
- Standard full pytest command.

## Risks and rollback

Risk is limited to stricter offline/pre-WebSocket diagnostics. Rollback is to
remove the dedicated violations while preserving the generic verifier failure.
