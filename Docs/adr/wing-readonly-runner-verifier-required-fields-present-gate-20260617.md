# WING Readonly Runner Verifier Required Fields Present Gate

## Context

The WING read-only preflight runner must validate the detached no-WING artifact
manifest verifier before any backend/WING WebSocket opens. The verifier report
has two related required-field signals: the explicit
`verifier_report_required_fields_missing` list and the boolean
`verifier_report_required_fields_present` check.

## Options considered

1. Keep only generic verifier failure and check-mismatch diagnostics.
2. Add a dedicated violation when the required-fields-present check is not
   `true`.

## Decision

Add
`prestart_summary_artifact_manifest_verifier_required_fields_present_flag_not_true`
when the detached verifier does not prove required report fields are present.

## Why this won

The runner already fails closed, but a false presence flag means the verifier
report is not audit-complete even if the missing-field list is empty or stale.
A dedicated violation gives the operator a concrete pre-WebSocket repair target.

## Rejected alternatives

Relying on generic verifier failure was rejected because it does not distinguish
a stale artifact mismatch from an incomplete verifier report shape.

## Implementation plan

- Extend `tools/wing_readonly_preflight_runner.py` with the dedicated
  violation.
- Add a pre-WebSocket regression test that mocks a false
  `verifier_report_required_fields_present` check while keeping the missing list
  empty.

## Test plan

- Focused runner verifier tests.
- Focused WING/no-WING tests.
- Standard full pytest command.

## Risks and rollback

Risk is limited to stricter offline/pre-WebSocket diagnostics. Rollback is to
remove the dedicated violation while preserving the generic verifier failure.
