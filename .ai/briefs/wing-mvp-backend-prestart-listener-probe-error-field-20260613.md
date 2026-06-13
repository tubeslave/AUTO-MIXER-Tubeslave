# Brief: WING MVP Backend Prestart Listener Probe Error Field

## Context

The active MVP blocker remains `WING read-only connection proof`. The no-WING
backend-port prestart gate already records `listener_process_error` when the
listener owner probe cannot identify the process, but the shared failure-detail
field contract did not require that field.

## Task

Promote `listener_process_error` into the backend-port prestart failure-detail
contract and keep the blocker, runbook, MVP docs, autoloop docs, generated
progress, and focused tests aligned.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure backend/WING listeners.
- Do not add live OSC/WING writes.
- Do not close the current blocker from documentation or offline evidence.

## Validation

- Focused no-WING release, handoff packet, WING MVP rails, Project Director,
  and direction guard tests.
- Project Director still selects `WING read-only connection proof`.
- Direction guard remains `on_target`.
- Standard backend pytest before commit.
