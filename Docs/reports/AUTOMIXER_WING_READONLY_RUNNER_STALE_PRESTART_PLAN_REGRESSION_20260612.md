# Automixer WING Read-Only Runner Stale Prestart Plan Regression Report

Date: 2026-06-12

## Scope

This slice strengthens the first supervised WING MVP blocker,
`WING read-only connection proof`, by proving the runner blocks stale expected
plans before any backend/WING WebSocket contact.

No WING Rack was contacted. No backend WING WebSocket session was opened. No
OSC/WING write was executed.

## Change

`tests/test_wing_readonly_preflight_runner.py` now covers a stale expected plan
whose `backend_port_prestart_check_failure_detail_fields` omits
`listener_confirmation_checklist`. The runner must return
`expected_plan_mismatch`, report
`expected_plan_backend_port_prestart_check_failure_detail_fields_mismatch`, and
send no requests.

## Verification

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py -k "expected_plan or plan_only"`
  - `4 passed, 23 deselected`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py`
  - `286 passed`
- `/opt/homebrew/bin/git diff --check`
  - PASS
- `python3 tools/memory/memory_lint.py`
  - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  - `3193 passed, 1 skipped, 1 warning`
