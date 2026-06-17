# WING Readonly Runner Verifier Required Field Gate

## Context

The read-only WING preflight runner checks the detached no-WING artifact
manifest verifier before opening the backend/WING WebSocket. The verifier also
declares `verifier_report_required_fields_missing`, which catches stale or
incomplete verifier report contracts.

## Options considered

1. Keep only the generic
   `prestart_summary_artifact_manifest_verifier_failed` violation.
2. Add a dedicated violation when required verifier report fields are missing.

## Decision

Add `prestart_summary_artifact_manifest_verifier_required_field_missing` when
`verifier_report_required_fields_missing` is non-empty.

## Why this won

Missing verifier report fields mean the pre-WebSocket artifact integrity gate is
stale. A dedicated violation makes the operator-visible blocker precise while
preserving the existing generic fail-closed reason.

## Rejected alternatives

Relying on the generic verifier failure was rejected because it hides contract
drift behind the same reason used for value mismatches and failed checks.

## Implementation plan

- Extend `tools/wing_readonly_preflight_runner.py` to emit the dedicated
  violation.
- Add a pre-WebSocket regression test that mocks a missing required verifier
  report field and asserts no WebSocket opens.

## Test plan

- Focused runner verifier tests.
- Focused WING/no-WING tests.
- Standard full pytest command.

## Risks and rollback

Risk is limited to stricter offline/pre-WebSocket validation. Rollback is to
remove the dedicated violation while keeping the generic verifier failure.
