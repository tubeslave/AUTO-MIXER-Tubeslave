# WING Readonly Runner Verifier Checks Object Gate

## Context

The WING read-only preflight runner consumes the detached no-WING artifact
manifest verifier before any backend/WING WebSocket opens. The verifier report
uses the `checks` object as the auditable boolean map for every required
manifest, summary, and authority-denial check.

## Options considered

1. Keep only the generic verifier failure and the expanded per-check mismatch
   list when `checks` is malformed.
2. Add a dedicated violation when `checks` is not an object.

## Decision

Add `prestart_summary_artifact_manifest_verifier_checks_not_object` when the
detached verifier report does not expose `checks` as an object.

## Why this won

Without a valid `checks` object, the runner cannot audit individual verifier
proofs before transport opens. A dedicated violation identifies the report shape
defect directly instead of burying it behind many individual check mismatches.

## Rejected alternatives

Relying on generic check mismatches was rejected because it hides the root cause
and makes operator repair less precise.

## Implementation plan

- Extend `tools/wing_readonly_preflight_runner.py` with the dedicated
  `checks` shape violation.
- Add a pre-WebSocket regression test that mocks a non-object `checks` field.

## Test plan

- Focused runner verifier tests.
- Focused WING/no-WING tests.
- Standard full pytest command.

## Risks and rollback

Risk is limited to stricter offline/pre-WebSocket diagnostics. Rollback is to
remove the dedicated violation while preserving the generic verifier failure.
