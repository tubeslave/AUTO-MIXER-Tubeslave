# Automixer WING Read-Only Preflight Listener Command-Line Diagnostics - 2026-06-11

## Summary

This slice strengthens the first supervised WING Rack MVP blocker, `WING read-only connection proof`.

When `no_wing_release_check --quick` finds a listener on the backend port, the listener record now includes the process command line when the OS exposes it. This helps the operator decide whether the listener is the intended Automixer backend before any real WING read-only preflight run.

## Boundary

- No WING Rack was contacted.
- No backend WebSocket WING session was opened.
- No local listener was stopped or reconfigured.
- No live OSC/WING writes were executed.
- No MVP blocker is closed by this diagnostic evidence.

## What Changed

- `tools/no_wing_release_check.py` now enriches listener process records with `command_line` from `ps -p <pid> -o command=`.
- If the command-line probe fails, the artifact records `command_line_error` while preserving the original `lsof` listener record.
- Tests cover both successful command-line enrichment and probe failure.

## Operator Impact

A busy-port artifact can now show whether the listener looks like the intended Automixer backend command, while still requiring manual confirmation. The prestart gate remains fail-closed and non-mutating.

## Verification

Focused verification:

```bash
python3 -m py_compile tools/no_wing_release_check.py
PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "listener_processes or backend_port"
```

Result: `5 passed, 183 deselected`.

Adjacent WING preflight/no-WING validation:

```bash
PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision_tool.py
```

Result: `252 passed`.

Full backend suite:

```bash
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
```

Result: `3179 passed, 1 skipped, 1 warning`.
