# Brief: WING MVP Supervised Gate Required Markers Next Action

Date: 2026-06-12

## Task

Promote `required_supervised_prestart_gate_markers` into WING MVP selected-action
surfaces.

## Context

The no-WING handoff summary already exposes the required supervised prestart
gate markers as a structured list. The selected action currently names the same
strings in the final operator step, but does not require the structured marker
list itself.

## Safety Boundary

This is documentation and test hardening only. It does not contact WING, does
not start a backend/WING session, does not stop local listeners, and does not
grant live authority.

## Done When

- The first WING MVP blocker selected action requires
  `required_supervised_prestart_gate_markers`.
- The read-only runbook, MVP doc, Project Director/autoloop docs, and generated
  progress report require the same field.
- Tests assert the field on operator-facing surfaces.
