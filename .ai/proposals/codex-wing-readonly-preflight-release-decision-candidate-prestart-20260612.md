# Codex Proposal: WING Read-Only Preflight Release Decision Candidate Prestart

## Short thesis

The no-WING release summary should include decision-ledger `candidate_reviews` so prestart safety fields remain visible at the release artifact boundary.

## What I understood

The project is closing the WING read-only connection proof path without widening live authority. Recent changes added candidate prestart fields to decision review output, but the release summary did not yet serialize or render those candidate rows.

## Proposed solution

- Copy `candidate_reviews` from the decision ledger check detail into `wing_readonly_preflight_review_decision_summary`.
- Render the candidate rows in the markdown release summary.
- Extend the existing summary regression test with a candidate row carrying backend-port prestart fields.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`

## Alternatives considered

- Leave this only in the decision review report. That keeps the release summary weaker and forces operators to cross-open multiple artifacts.
- Render a custom table. JSON is enough here because the surrounding release summary already uses compact serialized fields for ledger detail.

## Risks

- Large candidate rows can make markdown noisy. This is acceptable for a release gate artifact where auditability matters more than compactness.
- If future rows contain non-JSON-safe objects, summary generation could fail. Current ledger rows are plain JSON-compatible data.

## Test plan

- Focused no-WING release summary test.
- Affected WING/no-WING tests.
- Full backend test suite.
- Diff and memory lint.

## Where the other agent may disagree

Kimi may prefer a tabular markdown renderer for operator readability. I would keep the first step simple and lossless, then table selected fields later if the operator runbook needs it.
