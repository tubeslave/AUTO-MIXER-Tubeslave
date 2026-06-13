# Brief: WING MVP Backend Prestart Check Required By Packet Next Action

## Context

The first non-closed WING MVP blocker remains `WING read-only connection proof`.
The offline no-WING handoff packet already emits
`backend_port_prestart_check_required: true`, but the current blocker action and
autoloop docs do not yet require that top-level field before supervised proof
planning.

## Task

Promote `backend_port_prestart_check_required: true` into the WING MVP rails for
the current read-only handoff boundary.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure backend/WING listeners.
- Do not add live OSC/WING writes.
- Keep the change limited to docs, rails, council artifacts, and regression
  tests.

## Validation

- Focused Project Director and WING MVP rail tests.
- Project Director selected action still targets `WING read-only connection
  proof`.
- Direction guard remains `on_target`.
- Standard backend pytest before commit.
