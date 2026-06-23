# Codex Proposal: WING MVP Operator Markdown Required Field Lines Self-Check

## Short thesis

The no-WING operator Markdown gate should verify that its rendered
`operator_markdown_summary_required_field_lines` list contains every field in
the current Markdown contract.

## What I understood

The operator Markdown surface already checks that required field lines are
present as individual lines. It did not fail when the rendered required-lines
list itself was stale while those lines happened to be present elsewhere in the
Markdown.

## Proposed solution

Parse the rendered `operator_markdown_summary_required_field_lines` JSON list
from `summary.md`, compare it against
`NEXT_WING_PROOF_PLAN_OPERATOR_MARKDOWN_REQUIRED_FIELD_LINES`, expose
`operator_markdown_summary_required_field_lines_missing_members`, expose
`operator_markdown_summary_required_field_lines_match_contract`, and fail the
same operator Markdown surface check on mismatch.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`

## Alternatives considered

Only relying on individual field-line presence was rejected because it does not
prove the operator-visible required-lines contract was updated.

## Risks

The new parser may fail closed on malformed Markdown. That is intended for a
safety gate before real WING proof planning.

## Test plan

Run focused no-WING and Project Director tests, then no-WING quick and the full
test suite.

## Where the other agent may disagree

Kimi may argue the next highest-value step is physical WING capture. That is
true once hardware is available. Until then, this prevents stale operator
handoffs from pretending to satisfy the first blocker.
