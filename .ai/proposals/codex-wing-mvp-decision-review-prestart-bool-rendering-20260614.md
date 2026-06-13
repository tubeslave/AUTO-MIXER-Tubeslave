# Proposal: Preserve missing prestart booleans in decision review Markdown

## Short thesis

The decision review report should render candidate-prestart booleans without
turning missing evidence into safe-looking `false`.

## What I understood

The JSON ledger already validates accepted-candidate prestart trace fields. The
Markdown report is only a review surface, but if it renders missing fields as
`false`, an operator could mistake absent evidence for confirmed no-contact
evidence.

## Proposed solution

- Add a tiny Markdown boolean helper that returns `true`/`false` only for actual
  booleans and an empty value otherwise.
- Use it for candidate prestart contract booleans in
  `tools/review_wing_readonly_preflight_decisions.py`.
- Extend CLI tests so valid rows preserve false values and blocked incomplete
  rows leave missing fields empty.
- Add this report to WING MVP evidence.

## Likely files to touch

- `tools/review_wing_readonly_preflight_decisions.py`
- `tests/test_wing_readonly_preflight_review_decision_tool.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Keeping binary rendering was rejected because it can hide missing evidence.
Using JSON `null` text was rejected because the existing report style uses
compact inline values.

## Risks

The report may show empty values for malformed candidate rows. That is the
intended safer behavior; the blocker and ledger still explain why the candidate
is not valid.

## Test plan

- Focused decision review CLI test.
- Project Director/WING rail tests.
- no-WING quick release check.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may prefer explicit `missing`. I recommend the empty value because it keeps
the report style compact while preventing false evidence.
