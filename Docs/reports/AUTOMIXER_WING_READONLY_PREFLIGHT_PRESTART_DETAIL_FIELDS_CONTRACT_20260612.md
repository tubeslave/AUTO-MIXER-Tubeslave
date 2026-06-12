# Automixer WING Read-Only Preflight Prestart Detail Fields Contract - 2026-06-12

## Summary

This slice strengthens the first supervised WING Rack MVP blocker, `WING read-only connection proof`.

The shared WING preflight contract now requires the complete stable busy-port detail fields emitted by the no-WING prestart gate.

## Boundary

- No WING Rack was contacted.
- No backend WebSocket WING session was opened.
- No local listener was stopped or reconfigured.
- No live OSC/WING writes were executed.
- No MVP blocker is closed by this contract update.

## What Changed

`backend/wing_readonly_preflight_contract.py` now includes these required prestart failure detail fields:

- `listener_processes`
- `preflight_blocked_until_listener_confirmed`
- `intended_backend_confirmation_required`
- `safe_to_stop_listener_by_gate`
- `fail_closed_reason`
- `operator_next_step`

Tests now build expected WING preflight plans, handoff packets, and summary fixtures from the shared constant instead of duplicating the old two-field list.

## Verification

Focused validation:

```bash
PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision_tool.py
```

Result: `252 passed`.

After updating the autonomy readiness tests to the same shared contract:

```bash
PYTHONPATH=backend python3 -m pytest -q tests/test_autonomy_promotion_readiness.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision_tool.py
```

Result: `261 passed`.

Full backend suite:

```bash
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
```

Result: `3179 passed, 1 skipped, 1 warning`.
