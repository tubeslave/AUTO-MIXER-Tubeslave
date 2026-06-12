# Brief: WING MVP Execution Disallowed By Packet Next Action

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The no-WING handoff packet already carries `execution_allowed_by_packet: false`.

## Task

Make the selected action and operator-facing rails explicitly require:

- `execution_allowed_by_packet: false`

## Boundaries

- Do not contact WING.
- Do not grant live authority.
- Do not stop or reconfigure local listener processes.
- Do not execute any mixer action from a no-WING handoff packet.
- Keep this as a docs/tests/autoloop contract change around existing no-WING
  handoff evidence.

## Validation

- Regenerate MVP progress.
- Run focused WING rail tests.
- Run direction guard, Project Director, diff check, memory lint, and full
  pytest.
