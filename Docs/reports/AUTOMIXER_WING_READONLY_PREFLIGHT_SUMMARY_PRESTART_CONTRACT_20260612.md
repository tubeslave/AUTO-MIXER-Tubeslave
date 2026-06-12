# Automixer WING Read-Only Preflight Summary Prestart Contract - 2026-06-12

## Summary

This slice strengthens the first supervised WING Rack MVP blocker, `WING
read-only connection proof`.

Real `wing_readonly_preflight.v1` summary artifacts now carry the same
backend-port prestart contract as the offline expected plan, and evidence intake
rejects summaries whose prestart fields are missing or mismatched.

## Boundary

- No WING Rack was contacted.
- No backend WebSocket WING session was opened.
- No local listener was stopped or reconfigured.
- No live OSC/WING writes were executed.
- No MVP blocker is closed by this metadata/validation update.

## What Changed

- `tools/wing_readonly_preflight_runner.py` writes
  `backend_port_prestart_check_required`,
  `backend_port_prestart_check_must_pass_before_starting_backend`,
  `backend_port_prestart_check_command`, and
  `backend_port_prestart_check_failure_detail_fields` into summary artifacts.
- `backend/wing_readonly_preflight_evidence.py` validates those fields against
  the shared contract.
- Runner and evidence tests cover valid summaries and mismatched summary
  prestart metadata.

## Verification

Focused runner/evidence tests:

```bash
PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_evidence.py -k "prestart or physical_readbacks_without_write_messages or requires_expected_plan_before_websocket"
```

Result: `3 passed, 63 deselected`.

Adjacent WING/no-WING stack:

```bash
PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_autonomy_promotion_readiness.py
```

Result: `304 passed`.

Full backend suite:

```bash
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
```

Result: `3181 passed, 1 skipped, 1 warning`.

Static checks:

```bash
/opt/homebrew/bin/git diff --check
python3 tools/memory/memory_lint.py
```

Result: both passed.
