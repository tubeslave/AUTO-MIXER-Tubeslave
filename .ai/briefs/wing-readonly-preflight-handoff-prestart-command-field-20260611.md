# Brief: WING Read-Only Preflight Handoff Prestart Command Field

## Task

Make the WING read-only preflight handoff packet self-contained by surfacing the required prestart no-WING check command as a top-level safety field, and make no-WING artifact validation reject packets that omit or alter it.

## Context

The first supervised WING Rack MVP blocker is still `WING read-only connection proof`. The handoff packet already included `commands.prestart_backend_port_check`, but the top-level safety boundary only exposed the required flags and diagnostic fields. Operators and offline validators should be able to confirm the exact prestart command without depending on nested command parsing alone.

## Constraints

- Do not contact WING.
- Do not execute live OSC/WebSocket writes.
- Do not close any MVP blocker from offline evidence.
- Keep work scoped to the first non-closed WING MVP blocker.

## Test Plan

- `python3 -m py_compile tools/build_wing_readonly_preflight_handoff_packet.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py`
- Run adjacent WING preflight/no-WING checks before commit.
