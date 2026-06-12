# Automixer WING MVP Backend Prestart First-Step Next Action

Date: 2026-06-12

## Summary

The first open WING MVP blocker now explicitly requires the no-WING handoff
summary to show:

- `backend_port_prestart_check_operator_sequence_step_index: 0`
- `backend_port_prestart_check_must_be_first_operator_step: true`
- `backend_port_prestart_check_operator_sequence_step_is_first: true`

## Why

The backend-port prestart check must happen before any intended backend/WING
session. Project Director and autoloop docs now carry the same first-step
boundary as the read-only runbook and handoff validator.

## Changed

- `.ai/project-layer/wing_mvp_blockers.md` names the backend-port prestart
  first-step fields in the first blocker next action and current command.
- `Docs/reports/MVP_PROGRESS_LATEST.md` was regenerated from the blocker table.
- `tools/codex_autoloop/codex_cycle_prompt.md` and
  `tools/codex_autoloop/README.md` document the same pre-proof requirement.
- `tests/test_project_director_agent.py` and `tests/test_wing_mvp_rails.py`
  assert the first-step fields in selected-action and autoloop surfaces.

## Boundary

This report does not grant live authority. It does not contact WING, does not
start a backend/WING session, and does not close the read-only connection proof
blocker.

## Validation

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
  passed: `23 passed in 0.19s`.
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  reported `guard_status: on_target` and `guard_stop_the_line: false`.
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
  reported `status: on_target` and `stop_the_line: false`.
- `python3 tools/memory/memory_lint.py` passed.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` passed:
  `3205 passed, 1 skipped, 1 warning in 94.14s`.
