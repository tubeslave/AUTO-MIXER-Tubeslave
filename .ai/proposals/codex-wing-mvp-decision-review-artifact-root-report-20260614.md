# Codex Proposal: WING MVP Decision Review Artifact Root Report

## Short thesis

The decision-review report should show the same `artifact_root` that the
operator decision ledger validates.

## What I understood

The manual review decision ledger now rejects missing or stale `artifact_root`.
However, the decision-review report's candidate prestart section rendered
paths, expected artifacts, and backend-prestart fields without naming the
candidate root. That makes the root binding less visible to the operator and
to no-WING summary review.

## Proposed solution

Render `artifact_root` in
`tools/review_wing_readonly_preflight_decisions.py` candidate rows. Update the
decision-review focused test fixture to write runtime `artifact_root` into both
summary and decision record, and assert the field appears in report data and
Markdown. Update the no-WING summary fixture to preserve and assert the same
candidate root.

## Likely files to touch

- `tools/review_wing_readonly_preflight_decisions.py`
- `tests/test_wing_readonly_preflight_review_decision_tool.py`
- `tests/test_no_wing_release_check.py`
- Council/report docs for this slice.

## Alternatives considered

- Rely on raw `candidate_reviews` JSON in the summary. Rejected because the
  operator Markdown report is the human-facing review surface.
- Defer to ledger validation only. Rejected because downstream review should
  make the validated binding visible, not hidden.

## Risks

The change is report-only. Old decision-review fixtures without `artifact_root`
will show an empty field and fail the updated focused test when they are meant
to represent a valid accepted candidate.

## Test plan

- Focused decision-review tool tests.
- Full no-WING release-check module.
- Project Director rails, py_compile, memory lint, and `git diff --check`.

## Where the other agent may disagree

Kimi may argue raw JSON already carries the field. The Markdown report is the
operator-facing artifact, so duplicating the field there is useful and cheap.
