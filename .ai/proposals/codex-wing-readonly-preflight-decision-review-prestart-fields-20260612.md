# Proposal: WING Read-Only Preflight Decision Review Prestart Fields

## Short thesis

Add candidate prestart contract rows to the review decision ledger and decision
review report.

## What I understood

The operator decision report is the final offline check before any separate
manual readiness update. It should include the prestart safety boundary for the
candidate being accepted or rejected.

## Proposed solution

- Add `candidate_reviews` rows to the ledger summary.
- Pass those rows through `review_wing_readonly_preflight_decisions.py`.
- Render the prestart command and failure detail fields in markdown.
- Add tests for ledger rows and accepted decision markdown.

## Likely files to touch

- `backend/wing_readonly_preflight_review_decision.py`
- `tools/review_wing_readonly_preflight_decisions.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- `tests/test_wing_readonly_preflight_review_decision_tool.py`
- `.ai/briefs/...`
- `.ai/proposals/...`
- `Docs/adr/...`
- `Docs/reports/...`

## Alternatives considered

- Leave prestart fields only in artifact review.
- Re-open candidate summaries inside the decision review tool.

## Risks

The decision report grows slightly. This is acceptable because it makes the
final review artifact self-contained.

## Test plan

- Focused decision ledger/report tests.
- Adjacent WING/no-WING validation stack.
- `git diff --check`, memory lint, and full backend suite.

## Where the other agent may disagree

Kimi may argue the decision record should remain minimal. The decision report can
be richer without changing the operator decision record schema.
