# Brief: WING MVP Current Command Markdown Summary Surface

## Task

Make the Project Director / Current First Command rail explicitly require the
operator-readable no-WING `summary.md` proof-plan section to render the canonical
supervised proof-plan helper fields before supervised fader/gain proof planning.

## Context

The no-WING release check already renders
`required_supervised_proof_plan_summary_fields` and
`missing_required_supervised_proof_plan_summary_fields` in markdown. The next
Goal-mode slice should bind that operator-readable surface into the Project
Director baseline markers so a machine-only JSON contract is not enough for
handoff.

## Safety Boundary

This is documentation, autoloop rail, and test coverage only. It does not query
WING, mutate mixer state, grant live authority, or authorize fader/gain movement.

## Acceptance

- Current First Command says the operator-readable no-WING `summary.md` proof
  plan section must render the canonical helper fields.
- Project Director baseline coverage fails if that marker is removed.
- Autoloop prompt and README carry the same requirement.
- ADR/report record that WING remains blocked until real read-only proof and
  accepted operator review.
