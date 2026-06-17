# WING Readonly Runner Verifier Status Gate

## Context

The WING read-only preflight runner validates the detached no-WING artifact
manifest verifier before any backend/WING WebSocket opens. The verifier status
must be `passed` before the runner can proceed to transport.

## Options considered

1. Keep only the generic
   `prestart_summary_artifact_manifest_verifier_failed` violation.
2. Add a dedicated violation when the detached verifier status is not `passed`.

## Decision

Add `prestart_summary_artifact_manifest_verifier_status_not_passed` when the
verifier report status is anything other than `passed`.

## Why this won

A failed verifier status is the top-level reason the artifact bundle cannot be
trusted before WebSocket. It should remain visible even when `failed_checks` is
empty or malformed.

## Rejected alternatives

Relying on generic verifier failure was rejected because it does not distinguish
status failure from report value, shape, or individual check failures.

## Implementation plan

- Extend `tools/wing_readonly_preflight_runner.py` with the dedicated status
  violation.
- Add a pre-WebSocket regression test that mocks `status: failed` with no
  failed checks.

## Test plan

- Focused runner verifier tests.
- Focused WING/no-WING tests.
- Standard full pytest command.

## Risks and rollback

Risk is limited to stricter offline/pre-WebSocket diagnostics. Rollback is to
remove the dedicated status violation while preserving the generic verifier
failure.
