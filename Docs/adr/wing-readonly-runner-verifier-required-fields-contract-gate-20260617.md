# WING Readonly Runner Verifier Required Fields Contract Gate

## Context

The WING read-only preflight runner validates the detached no-WING artifact
manifest verifier before any backend/WING WebSocket opens. The verifier report
must prove that its required fields match the contract, not only that the
current missing-field list is empty.

## Options considered

1. Keep only generic verifier failure and check-mismatch diagnostics.
2. Add a dedicated violation when
   `verifier_report_required_fields_match_contract` is not `true`.

## Decision

Add
`prestart_summary_artifact_manifest_verifier_required_fields_contract_flag_not_true`
when the detached verifier does not prove its required-field set matches the
contract.

## Why this won

A false contract-match flag means the verifier report can be stale or
incomplete even when no individual missing fields are listed. The runner already
fails closed; the dedicated violation makes the pre-WebSocket repair target
operator-visible.

## Rejected alternatives

Relying on generic verifier failure was rejected because it does not distinguish
an artifact mismatch from a verifier report whose required-field contract proof
is invalid.

## Implementation plan

- Extend `tools/wing_readonly_preflight_runner.py` with the dedicated
  violation.
- Add a pre-WebSocket regression test that mocks
  `verifier_report_required_fields_match_contract` as false while keeping the
  missing-field list empty.

## Test plan

- Focused runner verifier tests.
- Focused WING/no-WING tests.
- Standard full pytest command.

## Risks and rollback

Risk is limited to stricter offline/pre-WebSocket diagnostics. Rollback is to
remove the dedicated violation while preserving the generic verifier failure.
