# ADR: WING Read-Only Preflight Backend Port Listener Confirmation

## Context

The WING read-only preflight runbook requires a no-WING quick check before the intended backend/WING session. If `127.0.0.1:8765` already has a listener, the gate must fail closed without stopping processes or pretending the listener is safe.

## Options considered

1. Keep only `listener_processes` and `operator_next_step`.
2. Add explicit machine-readable fail-closed fields for operator confirmation.
3. Probe the listener to decide whether it is the intended Automixer backend.

## Decision

Use option 2. Busy-port failures now include explicit confirmation and safety fields while preserving the existing listener diagnostics.

## Why this won

The no-WING gate should remain diagnostic and non-mutating. Machine-readable fields make downstream reports clearer without contacting the listener or changing local services.

## Rejected alternatives

Option 1 is less explicit for automation and reports. Option 3 may be useful later in a separate intended-backend confirmation tool, but it is outside the no-WING prestart boundary.

## Implementation plan

- Add busy-port detail fields in `tools/no_wing_release_check.py`.
- Extend tests for direct `check_port_free` output and release-check summaries.

## Test plan

- `python3 -m py_compile tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "backend_port or listener_processes"`
- Broader no-WING/WING focused tests.

## Risks and rollback

Risk: consumers that compare exact detail dictionaries may need updates. Rollback: remove the new detail fields and associated assertions. No live write path is changed.
