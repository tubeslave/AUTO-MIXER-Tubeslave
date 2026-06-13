# Codex Proposal: WING MVP Review Template Output Gate

## Short Thesis

The no-WING release check should validate the actual review decision template
output path, not only the presence of the output flag.

## What I Understood

The current MVP is still gated on real WING read-only proof. The pending review
decision template is an operator handoff artifact, so path drift can send the
operator to an unreviewed or stale JSON file. The existing guard proves that the
review command writes a template, but not that it writes the contract-owned
template path.

## Proposed Solution

- Parse the value after `--decision-template-output` from the review command.
- Compare it with `WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT`.
- Add no-WING detail and summary fields for the output path and match result.
- Fail closed with
  `wing_readonly_preflight_review_decision_template_output_mismatch` on drift.
- Cover missing-output and wrong-output cases in tests.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-review-template-output-gate-20260613.md`
- `.ai/proposals/codex-wing-mvp-review-template-output-gate-20260613.md`
- `Docs/adr/wing-mvp-review-template-output-gate-20260613.md`
- `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_TEMPLATE_OUTPUT_GATE_20260613.md`

## Alternatives Considered

- Keep flag-only validation. Rejected because a wrong path still passes.
- Move the comparison into the review tool only. Rejected because no-WING is the
  release gate and must catch drift before operator-facing readiness updates.
- Re-centralize constants. Rejected because the shared path already exists; the
  gap is validation, not ownership.

## Risks

The main risk is adding a stale fixture field to summary tests. Runtime risk is
low because this change does not run WING commands or alter command defaults.

## Test Plan

- Compile `tools/no_wing_release_check.py`.
- Run focused no-WING, Project Director, direction guard, and WING rails tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run `git diff --check`, memory lint, and the full pytest suite.

## Where The Other Agent May Disagree

Kimi may argue that path matching should also be enforced by the review CLI
itself. That is reasonable later, but the release gate still needs an explicit
fail-closed check because it is the operator-facing readiness boundary.
