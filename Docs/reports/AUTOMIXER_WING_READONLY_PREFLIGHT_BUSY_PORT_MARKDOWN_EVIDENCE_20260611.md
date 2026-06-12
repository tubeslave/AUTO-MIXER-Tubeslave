# Automixer WING Read-Only Preflight Busy-Port Markdown Evidence - 2026-06-11

## Summary

This slice strengthens the first supervised WING Rack MVP blocker, `WING read-only connection proof`.

When the required no-WING prestart gate fails because the backend port is busy, `summary.md` now explicitly shows the same confirmation fields as `summary.json`.

## Boundary

- No WING Rack was contacted.
- No backend WebSocket WING session was opened.
- No local listener was stopped or reconfigured.
- No live OSC/WING writes were executed.
- No MVP blocker is closed by this markdown evidence.

## What Changed

`tools/no_wing_release_check.py` now renders these fields in busy-port markdown details:

- `preflight_blocked_until_listener_confirmed`
- `intended_backend_confirmation_required`
- `safe_to_stop_listener_by_gate`
- `fail_closed_reason`

The markdown test also confirms listener `command_line` appears when present in the listener process record.

## Operator Impact

Operators reviewing `summary.md` can see directly that a busy backend port blocks real WING preflight until manual listener confirmation, and that the gate itself is not allowed to stop the listener.

## Verification

Focused verification:

```bash
python3 -m py_compile tools/no_wing_release_check.py
PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "summary_surfaces_backend_port_owner or backend_port or listener_processes"
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
