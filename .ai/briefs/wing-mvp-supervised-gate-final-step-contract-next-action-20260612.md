# Brief: WING MVP Supervised Gate Final-Step Contract Next Action

Date: 2026-06-12

## Task

Promote the structured `supervised_prestart_gate` final-step fields into the
WING MVP selected action, runbook, and autoloop instructions.

## Context

The no-WING handoff validator already requires `supervised_prestart_gate` to
match `expected_supervised_prestart_gate`. That structured gate includes
`operator_sequence_step_index: 9` and
`operator_sequence_step_must_be_final: true`, but the current selected-action
text only names the top-level final-step summary field.

## Safety Boundary

This is documentation and test hardening only. It does not contact WING, does
not start a backend/WING session, does not stop local listeners, and does not
grant live authority.

## Done When

- The first WING MVP blocker selected action names the structured supervised
  final-step fields.
- The read-only runbook, MVP doc, Project Director/autoloop docs, and generated
  progress report require the same fields.
- Tests assert those fields on the operator-facing surfaces.
