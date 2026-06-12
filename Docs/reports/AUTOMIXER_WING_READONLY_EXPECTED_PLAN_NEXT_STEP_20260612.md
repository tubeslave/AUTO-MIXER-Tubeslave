# Automixer WING Read-Only Expected Plan Next Step Report

Date: 2026-06-12

## Scope

This slice strengthens the first supervised WING MVP blocker,
`WING read-only connection proof`, by making blocked expected-plan summaries
actionable before any backend/WING WebSocket contact.

No WING Rack was contacted. No backend WING WebSocket session was opened. No
OSC/WING write was executed.

## Change

- `tools/wing_readonly_preflight_runner.py` now writes `operator_next_step`
  when the expected plan is missing, invalid, or mismatched.
- `tests/test_wing_readonly_preflight_runner.py` now verifies the next-step
  text and that those branches send no requests.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` now says blocked expected-plan
  summaries include `operator_next_step`.

## Verification

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py -k "expected_plan"`
  - `3 passed, 25 deselected`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py`
  - `28 passed`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py`
  - `287 passed`
- `/opt/homebrew/bin/git diff --check`
  - PASS
- `python3 tools/memory/memory_lint.py`
  - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  - `3194 passed, 1 skipped, 1 warning`
