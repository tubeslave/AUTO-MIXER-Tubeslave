# WING Readonly Runner Verifier Check Entry Flag Gate

## Context

The read-only WING preflight runner validates the detached no-WING artifact
manifest verifier before any backend/WING WebSocket opens. The verifier report
must prove that every failed check has a matching boolean entry in `checks`.
That proof is represented by the `failed_checks_have_check_entries` check.

## Options considered

1. Keep only the generic
   `prestart_summary_artifact_manifest_verifier_failed` violation.
2. Add a dedicated violation when `failed_checks_have_check_entries` is not
   `true`.

## Decision

Add
`prestart_summary_artifact_manifest_verifier_failed_check_entry_flag_not_true`
when the verifier report does not prove failed-check entry coverage.

## Why this won

A false coverage flag means the verifier report is not audit-complete even if
the missing-entry list is empty or malformed. The runner already blocks before
WebSocket; the dedicated violation makes the operator-visible reason precise.

## Rejected alternatives

Relying on the generic verifier failure was rejected because it hides the
difference between an artifact mismatch and an incomplete verifier report
coverage proof.

## Implementation plan

- Extend `tools/wing_readonly_preflight_runner.py` with the dedicated
  violation.
- Add a pre-WebSocket regression test that mocks the verifier coverage flag as
  false while keeping the missing-entry list empty.

## Test plan

- Focused runner verifier tests.
- Focused WING/no-WING tests.
- Standard full pytest command.

## Risks and rollback

Risk is limited to stricter offline/pre-WebSocket diagnostics. Rollback is to
remove the dedicated violation while preserving the generic verifier failure.
