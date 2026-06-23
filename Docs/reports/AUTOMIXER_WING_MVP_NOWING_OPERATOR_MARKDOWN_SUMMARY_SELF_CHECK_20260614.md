# WING MVP no-WING Operator Markdown Summary Self-Check

## Summary

The no-WING artifact bundle now records whether the operator-readable
`summary.md` proof-plan section rendered the required supervised proof-plan field
lines. Project Director Current First Command baseline coverage requires the
rendered verdict before supervised proof planning.

## Changed

- `tools/no_wing_release_check.py` adds an operator markdown surface self-check
  under `next_wing_proof_plan`.
- `summary.md` renders `operator_markdown_summary_required`,
  `operator_markdown_summary_field_lines_rendered`, and
  `operator_markdown_summary_missing_field_lines`.
- `tests/test_no_wing_release_check.py` covers both the rendered success path
  and the fail-closed missing-line path.
- Project Director rails require the rendered verdict in Current First Command,
  prompt, README, and rail tests.

## Safety Boundary

This is an offline artifact/reporting guard. It does not contact WING, query the
console, grant live authority, or authorize fader/gain movement.

## Remaining Blocker

The WING MVP remains blocked on a real WING read-only preflight capture and an
operator-reviewed accepted candidate before any supervised fader/gain proof run.
