# Brief: WING Read-Only Preflight Listener Command-Line Diagnostics

## Task

Improve the required no-WING prestart diagnostic for the first WING MVP blocker by including the busy backend listener's command line when the operating system exposes it.

## Context

`backend_port_free` can fail before the real WING read-only preflight because `127.0.0.1:8765` already has a listener. The artifact already reports PID, command name, and socket name. The operator still needs more evidence to decide whether the listener is the intended Automixer backend or an unexpected local service.

## Constraints

- Do not contact the listener over WebSocket.
- Do not stop or reconfigure any listener.
- Do not contact WING.
- Do not execute live OSC/WING writes.
- Do not close the MVP blocker from this diagnostic evidence.

## Test Plan

- `python3 -m py_compile tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "listener_processes or backend_port"`
- Run broader no-WING/WING focused tests before commit.
