# Automixer WING Read-Only Preflight Runbook Prestart Detail Fields - 2026-06-12

## Summary

This slice strengthens the first supervised WING Rack MVP blocker, `WING
read-only connection proof`.

The operator runbook now names the same stable busy-port failure detail fields as
the shared WING read-only preflight contract.

## Boundary

- No WING Rack was contacted.
- No backend WebSocket WING session was opened.
- No local listener was stopped or reconfigured.
- No live OSC/WING writes were executed.
- No MVP blocker is closed by this documentation/test update.

## What Changed

- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` now tells the operator to inspect
  `listener_processes`, `preflight_blocked_until_listener_confirmed`,
  `intended_backend_confirmation_required`, `safe_to_stop_listener_by_gate`,
  `fail_closed_reason`, and `operator_next_step` when `backend_port_free` fails.
- The runbook keeps `command_line` as optional diagnostics.
- `tests/test_no_wing_release_check.py` now verifies the runbook lists every
  shared prestart failure detail field.

## Verification

Focused runbook sync:

```bash
PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "runbook_lists_prestart_failure_detail_fields or prestart_contract_uses_shared_backend_constants"
```

Result: `2 passed, 187 deselected`.

Adjacent WING/no-WING stack:

```bash
PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_autonomy_promotion_readiness.py
```

Result: `262 passed`.

Full backend suite:

```bash
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
```

Result: `3180 passed, 1 skipped, 1 warning`.

Static checks:

```bash
/opt/homebrew/bin/git diff --check
python3 tools/memory/memory_lint.py
```

Result: both passed.
