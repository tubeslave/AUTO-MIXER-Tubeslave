# ADR: WING MVP Review Decision Template-Status Unedited Gate

## Context

The WING read-only decision template includes a pending `template_status` value:
`operator_must_replace_placeholders_before_ledger_acceptance`. The ledger could
accept a record if the operator changed the main decision fields but left this
stale pending marker in place.

## Options Considered

1. Ignore `template_status` in the ledger.
2. Reject records that keep the generated pending `template_status`.
3. Require a new explicit operator-reviewed `template_status`.

## Decision

Use option 2. The ledger rejects records that still carry the generated pending
template marker.

## Why This Won

The marker says the template is not ready for ledger acceptance. Treating it as
diagnostic-only would let incomplete manual review state pass.

## Rejected Alternatives

Ignoring the marker was rejected because it weakens the manual review boundary.
Requiring a new replacement status was rejected because the current schema does
not define one.

## Implementation Plan

- Add the shared template status constant to ledger validation.
- Emit `review_decision_template_status_unedited` for stale pending markers.
- Add test and runbook coverage.

## Test Plan

- Run targeted ledger and runbook rails tests.
- Run focused WING/no-WING tests.
- Run full standard pytest.
- Run no-WING quick, memory lint, Project Director, and `git diff --check`.

## Risks And Rollback

Risk is limited to invalidating stale pending-template records. Rollback is
removing the marker check, but that would restore the stale accepted-decision
path.
