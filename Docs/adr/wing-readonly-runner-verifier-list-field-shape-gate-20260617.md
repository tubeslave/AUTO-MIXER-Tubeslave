# WING Readonly Runner Verifier List Field Shape Gate

## Context

The WING read-only preflight runner consumes the detached no-WING artifact
manifest verifier before any backend/WING WebSocket opens. Several verifier
fields are list-shaped diagnostics: `failed_checks`,
`failed_checks_missing_check_entries`, and
`verifier_report_required_fields_missing`.

## Options considered

1. Keep only the generic verifier failure when these fields are malformed.
2. Add dedicated violations when the verifier returns non-list values for
   list-shaped diagnostics.

## Decision

Add dedicated pre-WebSocket violations for malformed verifier list fields:

- `prestart_summary_artifact_manifest_verifier_failed_checks_not_list`
- `prestart_summary_artifact_manifest_verifier_failed_checks_missing_check_entries_not_list`
- `prestart_summary_artifact_manifest_verifier_required_fields_missing_not_list`

## Why this won

A malformed verifier report cannot be audited safely before transport opens.
The runner already fails closed; dedicated violations make the exact verifier
shape defect operator-visible.

## Rejected alternatives

Relying on generic verifier failure was rejected because it hides report-shape
defects behind the same message used for ordinary artifact mismatches.

## Implementation plan

- Extend `tools/wing_readonly_preflight_runner.py` with list-shape violations.
- Add a pre-WebSocket regression test that mocks malformed verifier list fields.

## Test plan

- Focused runner verifier tests.
- Focused WING/no-WING tests.
- Standard full pytest command.

## Risks and rollback

Risk is limited to stricter offline/pre-WebSocket diagnostics. Rollback is to
remove the dedicated violations while preserving the generic verifier failure.
