# Proposal: WING MVP Review Decision Review-ID Required Gate

## Short Thesis

The manual review-decision ledger should reject accepted records that omit the
operator-facing `review_id`.

## What I Understood

The WING MVP remains blocked until a real read-only preflight capture has a
clean operator review. The accepted-decision template includes `review_id`, but
the ledger currently accepts a missing field by using an internal fallback ID.

## Proposed Solution

- Detect a missing explicit `review_id` in the ledger validator.
- Add `review_decision_review_id_missing` to record violations.
- Add a regression test proving the record is invalid and not accepted.
- Update the runbook wording and rails test so operators keep `review_id`.

## Likely Files To Touch

- `backend/wing_readonly_preflight_review_decision.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tests/test_wing_mvp_rails.py`

## Alternatives Considered

- Require `review_id` to match `review-wing-readonly-<run_id>`. Rejected for
  now because existing tests and manual replacement flows allow custom IDs.
- Keep the fallback-only behavior. Rejected because it weakens the required
  manual decision identity.

## Risks

Risk is low. The change may invalidate older hand-written decision records that
omitted `review_id`, but those records did not satisfy the required template
contract.

## Test Plan

- Targeted ledger and runbook rails tests.
- Focused WING/no-WING tests.
- Full standard pytest.
- no-WING quick, memory lint, Project Director, and `git diff --check`.

## Where The Other Agent May Disagree

Kimi may prefer strict generated-ID matching. I would defer that until there is
evidence that custom review IDs cause ambiguity; this step closes the concrete
missing-field hole without breaking custom operator naming.
