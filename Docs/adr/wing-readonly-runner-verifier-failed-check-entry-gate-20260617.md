# WING Readonly Runner Verifier Failed Check Entry Gate

## Context

The read-only WING preflight runner must verify the detached no-WING artifact
manifest bundle before opening a backend/WING WebSocket. The verifier report
includes both `failed_checks` and `failed_checks_missing_check_entries`; the
second field proves every failed check has a corresponding boolean entry under
`checks`.

## Options considered

1. Keep only the generic
   `prestart_summary_artifact_manifest_verifier_failed` violation.
2. Add a dedicated violation when failed checks lack check entries.

## Decision

Add `prestart_summary_artifact_manifest_verifier_failed_check_entry_missing`
when `failed_checks_missing_check_entries` is non-empty.

## Why this won

A missing failed-check entry means the verifier report is incomplete and cannot
be audited precisely before transport opens. A named violation preserves the
generic fail-closed behavior while making the operator-visible reason concrete.

## Rejected alternatives

Relying on the generic verifier failure was rejected because it does not
distinguish a real artifact mismatch from an incomplete verifier report shape.

## Implementation plan

- Extend `tools/wing_readonly_preflight_runner.py` to emit the dedicated
  violation.
- Add a pre-WebSocket regression test that mocks a verifier report with
  `failed_checks_missing_check_entries`.

## Test plan

- Focused runner verifier tests.
- Focused WING/no-WING tests.
- Standard full pytest command.

## Risks and rollback

Risk is limited to stricter offline/pre-WebSocket validation. Rollback is to
remove the dedicated violation while preserving the generic verifier failure.
