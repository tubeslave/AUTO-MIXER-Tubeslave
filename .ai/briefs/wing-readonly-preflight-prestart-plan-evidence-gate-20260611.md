# Brief: WING Read-Only Preflight Prestart Plan Evidence Gate

## Task

Strengthen the first supervised WING Rack MVP blocker, `WING read-only connection proof`, by making the offline artifact evidence review reject expected-plan files that omit or alter the required prestart no-WING release check boundary.

## Context

`Docs/runbooks/WING_READONLY_PREFLIGHT.md` requires `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` before the intended backend/WING session starts. `tools/wing_readonly_preflight_runner.py --plan-only` already records that boundary in the expected plan, but the file-backed evidence validator did not independently reject a tampered expected plan that removed or replaced the prestart check fields.

## Constraints

- Do not contact WING.
- Do not execute live OSC/WebSocket writes.
- Do not close any MVP blocker from offline evidence.
- Keep this limited to the first non-closed MVP blocker.

## Test Plan

- `python3 -m py_compile backend/wing_readonly_preflight_evidence.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py`
- Run adjacent WING preflight/no-WING tests before commit.
