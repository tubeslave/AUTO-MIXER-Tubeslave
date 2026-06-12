# Brief: WING MVP Supervised Gate Decision Runbook Contract Next Action

Date: 2026-06-12

## Task

Promote the structured `supervised_prestart_gate` accepted-decision and runbook
fields into WING MVP selected-action surfaces.

## Context

The no-WING handoff validator already requires `supervised_prestart_gate` to
match `expected_supervised_prestart_gate`. That structured gate includes
`requires_operator_decision_status: accepted_decision_ready` and
`unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`, but the
operator-facing next-action text did not name those fields explicitly.

## Safety Boundary

This is documentation and test hardening only. It does not contact WING, does
not start a backend/WING session, does not stop local listeners, and does not
grant live authority.

## Done When

- The first WING MVP blocker selected action names the structured accepted
  decision and runbook fields.
- The read-only runbook, MVP doc, Project Director/autoloop docs, and generated
  progress report require the same fields.
- Tests assert those fields on the operator-facing surfaces.
