# Proposal: WING MVP Review Decision Template Required-Edits Surface

## Short Thesis

The operator-facing review report must list all edits required before a pending
decision template can become a ledger-acceptable decision.

## What I Understood

The WING MVP remains blocked until a real read-only capture is manually
reviewed. The ledger rejects stale pending `template_status`, but the generated
review report did not yet surface that required edit.

## Proposed Solution

- Add `review_decision_template_required_edits` to the review report.
- Render the required edits in the markdown Review Decision Template section.
- Update CLI help for `--decision-template-output`.
- Add focused tests for JSON/markdown/report behavior and stale-template
  violation visibility.

## Likely Files To Touch

- `tools/review_wing_readonly_preflight_artifacts.py`
- `tests/test_wing_readonly_preflight_review_tool.py`

## Alternatives Considered

- Rely on the runbook only. Rejected because the generated report is what the
  operator has open during a hardware review.
- Add a new accepted operator status. Rejected because the current ledger only
  needs to reject the known pending marker.

## Risks

Risk is low. This is report/help text and test coverage; runtime WING behavior
does not change.

## Test Plan

- Targeted review tool tests.
- Focused WING/no-WING tests.
- Full standard pytest.
- no-WING quick, memory lint, Project Director, and `git diff --check`.

## Where The Other Agent May Disagree

Kimi may ask for schema-level required-edits constants in the shared contract.
I would keep this in the review tool for now because it is generated-report
presentation, while the ledger remains the enforcement point.
