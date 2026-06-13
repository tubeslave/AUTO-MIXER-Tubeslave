# No-WING Release Check

- generated_at: `2026-06-13T09:01:19Z`
- status: `failed`
- connection_policy: `no_hardware_no_network_no_live_writes`
- live_write: `false`

## Checks

- `backend_port_free`: `failed`
  - listener_processes: `[{"command": "Python", "command_line": "/Library/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python main.py", "name": "*:8765", "pid": 87210}]`
  - listener_confirmation_checklist: `["confirm whether the listener is the intended Automixer backend for this WING preflight", "confirm the listener command line and PID match the operator-started backend session", "do not stop or reconfigure the listener from this no-WING gate", "rerun the no-WING gate after manual confirmation or after clearing an unexpected listener"]`
  - preflight_blocked_until_listener_confirmed: `True`
  - intended_backend_confirmation_required: `True`
  - safe_to_stop_listener_by_gate: `False`
  - fail_closed_reason: `backend_port_listener_requires_operator_confirmation`
  - operator_next_step: `Stop or reconfigure the unexpected local listener only after confirming it is not the intended Automixer backend, then rerun the no-WING gate and WING read-only preflight.`

## Failed Checks

- `backend_port_free`
