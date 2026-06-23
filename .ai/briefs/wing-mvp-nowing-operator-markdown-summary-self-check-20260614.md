# Brief: WING MVP no-WING Operator Markdown Summary Self-Check

## Task

Make the no-WING proof-plan summary surface self-check explicit and auditable:
`summary.json` and `summary.md` should show that the operator-readable
`summary.md` proof-plan section rendered the required supervised proof-plan
field lines.

## Context

Project Director now requires the operator-readable no-WING `summary.md` proof
plan section. The next gap is making the generated no-WING artifacts carry a
machine-readable verdict for that surface, so future markdown drift fails
closed instead of silently preserving only JSON fields.

## Safety Boundary

This is offline artifact generation and test coverage only. It does not contact
WING, query the console, grant live authority, or authorize fader/gain movement.

## Acceptance

- `next_wing_proof_plan` summaries include operator markdown surface fields.
- `summary.md` renders the same operator markdown surface verdict.
- Missing required markdown field lines make the summary fail closed.
- Project Director baseline coverage requires the rendered verdict before
  supervised proof planning.
