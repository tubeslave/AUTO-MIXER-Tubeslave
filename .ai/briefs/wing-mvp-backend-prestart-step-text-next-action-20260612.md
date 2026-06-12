# Brief: WING MVP Backend Prestart Step Text Next Action

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The no-WING handoff validator already exposes the first operator step text as
`backend_port_prestart_check_operator_sequence_step`.

## Task

Make the selected action and operator-facing rails explicitly require that the
first operator step text includes:

- `prestart_backend_port_check`
- `unexpected listeners`

## Boundaries

- Do not contact WING.
- Do not grant live authority.
- Do not stop or reconfigure local listener processes.
- Keep this as a docs/tests/autoloop contract change around the existing
  no-WING handoff evidence.

## Validation

- Regenerate MVP progress.
- Run focused WING rail tests.
- Run direction guard, Project Director, diff check, memory lint, and full
  pytest.
