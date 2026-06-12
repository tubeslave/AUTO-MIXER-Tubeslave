# Brief: WING MVP Backend Prestart Required Markers Next Action

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The no-WING handoff already carries
`backend_port_prestart_gate.required_markers` in the structured backend-port
prestart gate.

## Task

Make the selected action and operator-facing rails explicitly require
`backend_port_prestart_gate.required_markers` to exactly list:

- `Before starting the intended backend/WING session`
- `prestart_backend_port_check`
- `unexpected listeners`

## Boundaries

- Do not contact WING.
- Do not grant live authority.
- Do not stop or reconfigure local listener processes.
- Keep this as a docs/tests/autoloop contract change around existing no-WING
  handoff evidence.

## Validation

- Regenerate MVP progress.
- Run focused WING rail tests.
- Run direction guard, Project Director, diff check, memory lint, and full
  pytest.
