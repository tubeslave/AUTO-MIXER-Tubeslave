# ADR: WING read-only runner keeps verifier list-shape diagnostics separate

## Context

The read-only preflight runner blocks before opening a backend/WING WebSocket
when the detached no-WING artifact manifest verifier fails. Some verifier
fields are list contracts:

- `failed_checks`
- `failed_checks_missing_check_entries`
- `verifier_report_required_fields_missing`

Malformed values for those fields are a different failure from valid but
non-empty lists.

## Options considered

1. Treat any truthy value as both malformed and non-empty.
2. Report non-empty-list violations only when the value is actually a list.

## Decision

Use option 2. The runner now emits list-shape violations for malformed values
without also claiming that the malformed value is a non-empty list.

## Why this won

Operator review needs precise failure causes. A string like `"not-a-list"` must
block the preflight as a report-shape failure, but it should not be reported as
an actual failed-check list or missing-required-field list.

## Rejected alternatives

Keeping truthiness-based reporting was too imprecise. It could send the operator
toward inspecting failed check contents when the real problem is that the
verifier report schema drifted.

## Implementation plan

Use `isinstance(value, list) and value` before adding non-empty-list
violations. Keep the existing fail-closed behavior for malformed values through
the `*_not_list` violations.

## Test plan

Extend the malformed verifier list fields regression to assert that malformed
values block before WebSocket without also surfacing non-empty-list violations.
Run the targeted regression, focused WING/no-WING suite, no-WING quick gate,
`git diff --check`, and a full-equivalent test pass.

## Risks and rollback

Risk is low because malformed values still fail closed. Rollback is to restore
truthiness checks, but that would reintroduce ambiguous operator diagnostics.
