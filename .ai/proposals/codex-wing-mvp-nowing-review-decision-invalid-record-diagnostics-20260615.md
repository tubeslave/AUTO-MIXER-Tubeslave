# Proposal: WING MVP no-WING Review-Decision Invalid-Record Diagnostics

## Short Thesis

The no-WING release summary should mirror the decision-review invalid-record
diagnostics already available in the standalone report.

## What I Understood

The WING MVP remains blocked on real read-only evidence plus manual operator
review. Operators use both the standalone decision-review report and the
no-WING quick summary; both surfaces must explain why a review decision is
invalid without granting any readiness authority.

## Proposed Solution

- Normalize invalid review decision records from the ledger records inside
  `tools/no_wing_release_check.py`.
- Add `invalid_records` to the schema-check detail.
- Propagate the field into `wing_readonly_preflight_review_decision_ledger`
  summary JSON.
- Render it in `summary.md`.
- Add targeted tests for validator detail and summary markdown.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- WING runbook and Council docs

## Alternatives Considered

- Keep only `invalid_record_ids`. Rejected because the operator still has to
  inspect nested ledger records to find the stale `template_status` cause.
- Recompute violations in no-WING. Rejected because the ledger already owns
  validation; no-WING should only surface the ledger result.

## Risks

Risk is limited to report shape and markdown output. The existing fail-closed
condition for invalid records remains unchanged.

## Test Plan

- Targeted no-WING validator/summary tests.
- Focused WING/no-WING suite.
- Full standard pytest.
- no-WING quick, diff check, memory lint, and Project Director.

## Where The Other Agent May Disagree

Kimi may prefer making invalid record diagnostics a shared contract constant.
I would first mirror the ledger records because this is a report propagation
gap, not a new validation rule.
