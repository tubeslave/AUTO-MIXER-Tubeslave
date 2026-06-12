# Brief: WING MVP Backend Prestart Check Command Next Action

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The no-WING handoff already carries `backend_port_prestart_check_command` as a
structured command field.

## Task

Make the selected action and operator-facing rails explicitly require
`backend_port_prestart_check_command` to exactly match:

- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`

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
