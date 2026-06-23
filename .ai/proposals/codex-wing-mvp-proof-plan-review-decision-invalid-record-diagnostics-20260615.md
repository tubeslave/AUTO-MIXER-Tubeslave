# Proposal: WING MVP Proof-Plan Review-Decision Invalid-Record Diagnostics

## Short Thesis

The supervised proof plan should carry the same invalid review-decision record
diagnostics that the ledger and no-WING review-decision summary already expose.

## What I Understood

The WING MVP remains blocked on real read-only WING evidence and manual
operator review. The next proof-plan section is an operator-facing surface, so
it should not reduce invalid review decisions to counts when the ledger knows
the exact stale `template_status` and violation list.

## Proposed Solution

- Normalize invalid review-decision records in
  `backend/supervised_wing_proof_plan.py`.
- Add `readonly_preflight_review_decision_invalid_records` to the required
  proof-plan summary field contract.
- Make `tools/no_wing_release_check.py` validate this field against the ledger
  and carry it into the aggregated `next_wing_proof_plan` summary.
- Render the field in no-WING `summary.md`.
- Add regression tests for proof-plan output, no-WING summary output, required
  field coverage, and stale invalid-record mismatch detection.

## Likely Files To Touch

- `backend/supervised_wing_proof_plan.py`
- `backend/wing_readonly_preflight_contract.py`
- `tools/no_wing_release_check.py`
- `tests/test_supervised_wing_proof_plan.py`
- `tests/test_no_wing_release_check.py`
- Council docs and report index

## Alternatives Considered

- Keep only invalid record counts and ids. Rejected because the operator still
  has to inspect nested ledger state to find the stale-template cause.
- Recompute invalid records in no-WING. Rejected because the review-decision
  ledger owns validation; proof-plan and no-WING should project that result.

## Risks

Risk is limited to report shape and test fixtures. The change intentionally
adds a fail-closed mismatch diagnostic if the proof plan drifts from ledger
invalid records.

## Test Plan

- Targeted proof-plan and no-WING tests.
- Focused WING/no-WING suite.
- Standard pytest.
- no-WING quick, diff check, memory lint, and Project Director guard.

## Where The Other Agent May Disagree

Kimi may prefer a broader shared schema object for all review-decision
diagnostics. I would keep this patch narrow: add only the field needed to stop
the proof-plan surface from hiding invalid records.
