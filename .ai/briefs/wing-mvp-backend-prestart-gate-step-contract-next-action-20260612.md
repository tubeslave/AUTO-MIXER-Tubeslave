# Brief: WING MVP Backend Prestart Gate Step Contract Next Action

Date: 2026-06-12

## Task

Promote the structured `backend_port_prestart_gate` first-step fields into the
WING MVP selected action and autoloop instructions.

## Context

The no-WING handoff validator already requires `backend_port_prestart_gate` to
match `expected_backend_port_prestart_gate`. That structured gate includes
`operator_sequence_step_index: 0` and
`operator_sequence_step_must_be_first: true`, but the current blocker next-action
text only names the upper-level first-step summary fields.

## Safety Boundary

This is documentation and test hardening only. It does not contact WING, does
not start a backend/WING session, does not stop local listeners, and does not
grant live authority.

## Done When

- The first WING MVP blocker selected action names the structured prestart gate
  first-step fields.
- Project Director/autoloop docs require the same fields.
- Tests assert those fields on the operator-facing surfaces.
- The generated MVP progress report is refreshed from the blocker table.
