# Brief: WING MVP Backend Prestart Failure Detail Fields Next Action

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The no-WING handoff already carries
`backend_port_prestart_check_failure_detail_fields` as the structured list of
fields operators must review when the backend-port prestart check detects
unexpected listeners.

## Task

Make the selected action and operator-facing rails explicitly require
`backend_port_prestart_check_failure_detail_fields` to exactly list:

- `listener_processes`
- `listener_confirmation_checklist`
- `preflight_blocked_until_listener_confirmed`
- `intended_backend_confirmation_required`
- `safe_to_stop_listener_by_gate`
- `fail_closed_reason`
- `operator_next_step`

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
