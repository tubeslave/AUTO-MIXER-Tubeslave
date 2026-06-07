# Automixer WING Read-Only Preflight Request-ID Correlation Gate - 2026-06-07

## Summary

This slice strengthens the real WING read-only preflight proof contract.

Every runner-generated `query_mixer_parameter` request now carries a stable
`request_id` such as `wing-readonly-preflight:ch3:fader`. The backend echoes
that id in both successful readback responses and blocked read responses. The
runner and offline evidence validator fail closed if the response request id is
missing or mismatched.

## Safety Boundary

- No WING Rack session was opened.
- No OSC/WebSocket hardware transport was contacted.
- No fader, gain, trim, EQ, dynamics, routing, mute, FX, scene, matrix, or DCA
  write path was widened.
- No readiness matrix row or MVP blocker was closed.
- The next real action remains `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.

## Files Changed

- `backend/handlers/mixer_handlers.py`
- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tools/no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- focused tests for runner, evidence, review, no-WING, and backend handler behavior
- `.ai/briefs/wing-readonly-preflight-request-id-correlation-gate-20260607.md`
- `.ai/proposals/codex-wing-readonly-preflight-request-id-correlation-gate-20260607.md`
- `Docs/adr/wing-readonly-preflight-request-id-correlation-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REQUEST_ID_CORRELATION_GATE_20260607.md`
- `Docs/INDEX.md`

## Validation

- `python3 -m py_compile tools/wing_readonly_preflight_runner.py backend/wing_readonly_preflight_evidence.py backend/handlers/mixer_handlers.py tools/no_wing_release_check.py` - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_no_wing_release_check.py tests/test_autonomy_promotion_readiness.py tests/test_server.py::TestWingDeploymentWriteBoundary::test_query_mixer_parameter_reads_gain_without_writing tests/test_server.py::TestWingDeploymentWriteBoundary::test_query_mixer_parameter_blocks_unsupported_address` - `243 passed`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py` - PASS; status `on_target`, score `90`, `drift_item_count=0`, `mixed_item_count=0`
- `python3 tools/memory/memory_lint.py` - PASS
- `/opt/homebrew/bin/git diff --check` - PASS
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_request_id_correlation_20260607` - PASS; top-level status `passed`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - `3001 passed, 1 skipped, 1 warning`

## Current Boundary

This improves proof quality for the first WING blocker. It does not prove a real
WING connection by itself and does not authorize supervised writes.
