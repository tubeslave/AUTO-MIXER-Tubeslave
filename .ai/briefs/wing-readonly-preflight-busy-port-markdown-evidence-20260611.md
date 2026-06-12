# Brief: WING Read-Only Preflight Busy-Port Markdown Evidence

## Task

Make the no-WING prestart summary markdown explicitly show busy-port confirmation fields for the first WING MVP blocker.

## Context

The JSON artifact already records that a busy backend port blocks real WING preflight until the operator confirms whether the listener is the intended Automixer backend. The markdown summary should expose the same fail-closed reason and safety flags so the operator-visible report is not less clear than the machine-readable artifact.

## Constraints

- Do not contact WING.
- Do not probe the local backend WebSocket.
- Do not stop or reconfigure local listeners.
- Do not close any MVP blocker from this offline evidence.

## Test Plan

- `python3 -m py_compile tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "summary_surfaces_backend_port_owner or backend_port or listener_processes"`
- Run broader no-WING/WING focused tests before commit.
