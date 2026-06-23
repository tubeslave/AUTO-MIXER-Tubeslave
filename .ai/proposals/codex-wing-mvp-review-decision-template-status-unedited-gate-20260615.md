# Proposal: WING MVP Review Decision Template-Status Unedited Gate

## Short Thesis

The manual review-decision ledger should reject records that still carry the
generated pending `template_status` marker.

## What I Understood

The WING MVP remains blocked until a real read-only preflight capture has a
manual accepted decision. The generated decision template includes an explicit
pending marker. The ledger already rejects untouched templates through
`decision`, `reviewed_by`, and `reviewed_at`, but an operator could edit those
fields and accidentally leave the stale pending marker.

## Proposed Solution

- Import the shared pending decision-template status into the ledger.
- Add `review_decision_template_status_unedited` when that marker remains.
- Include `template_status` in the validated record diagnostics.
- Add a regression test and runbook wording.

## Likely Files To Touch

- `backend/wing_readonly_preflight_review_decision.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tests/test_wing_mvp_rails.py`

## Alternatives Considered

- Reject every non-empty `template_status`. Rejected because a future explicit
  operator-reviewed status may be useful.
- Ignore the field. Rejected because the existing marker explicitly says the
  template still needs operator replacement before acceptance.

## Risks

Risk is low. The change invalidates stale template-derived records that still
carry the pending marker, which is the intended fail-closed behavior.

## Test Plan

- Targeted ledger and runbook rails tests.
- Focused WING/no-WING tests.
- Full standard pytest.
- no-WING quick, memory lint, Project Director, and `git diff --check`.

## Where The Other Agent May Disagree

Kimi may prefer requiring an explicit replacement status. I would only reject
the known stale pending marker in this step to avoid inventing a new operator
field contract.
