# Proposal: WING MVP Decision Review Invalid-Record Diagnostics

## Short Thesis

The decision-review report should show the ledger's invalid-record diagnostics
without requiring an operator to inspect nested JSON.

## What I Understood

The WING MVP remains blocked on real read-only evidence and manual review. The
ledger correctly rejects stale pending decision templates, but the standalone
report only surfaced invalid ids at the top level.

## Proposed Solution

- Add a top-level `invalid_records` list to
  `tools/review_wing_readonly_preflight_decisions.py`.
- Include `review_id`, `artifact_source_id`, `decision`, `template_status`,
  `violations`, and `unsafe_authority_flags`.
- Render the list in the decision-review markdown.
- Add a focused test that accepted records with the generated pending
  `template_status` show `review_decision_template_status_unedited`.

## Likely Files To Touch

- `tools/review_wing_readonly_preflight_decisions.py`
- `tests/test_wing_readonly_preflight_review_decision_tool.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- Council docs and report index entries

## Alternatives Considered

- Rely on `ledger.records` only. Rejected because the markdown report is the
  operator-facing artifact during review.
- Add only a special-case `template_status` field. Rejected because duplicate,
  unsafe-authority, and other invalid decisions need the same diagnostic shape.

## Risks

Risk is limited to report shape and markdown output. Runtime WING behavior and
readiness authority are unchanged.

## Test Plan

- Targeted decision-review diagnostic test.
- Focused WING/no-WING tests.
- Full standard pytest.
- no-WING quick, memory lint, Project Director, and `git diff --check`.

## Where The Other Agent May Disagree

Kimi may prefer adding an `invalid_records` summary directly to the ledger
summary. I would keep the ledger unchanged because it already returns complete
records, and this is a presentation/reporting gap.
