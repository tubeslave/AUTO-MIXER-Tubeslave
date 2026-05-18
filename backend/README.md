# Backend Source

This folder contains backend source code and runtime-adjacent manual probes.

Full backend documentation is in `Docs/backend/`.

## Canonical Runtime

- `server.py`
- `wing_client.py`
- `handlers/`
- `operator_*.py`
- `auto_soundcheck_engine.py`
- `replay_write_intent_adapter.py`

## Manual Probes

Files named `test_*.py`, `check_*.py`, `query_*.py`, `route_*.py`,
`find_*.py`, `list_*.py`, and `monitor_*.py` at this level are manual probes or
legacy checks, not the canonical automated test suite.

Canonical automated tests live in `tests/`.

Manual probe inventory: `Docs/backend/MANUAL_PROBES.md`.

## Lab And Quarantine

Use `backend/lab_only/` for legacy, quarantine, and experimental scripts that
must not be imported by live runtime.
