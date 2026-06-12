# Automixer WING MVP Backend Prestart Step Text Next Action

Date: 2026-06-12

## Summary

The first open WING MVP blocker now explicitly requires the no-WING handoff
summary to show `backend_port_prestart_check_operator_sequence_step` including:

- `prestart_backend_port_check`
- `unexpected listeners`

## Why

The handoff already requires the backend-port prestart check to be the first
operator step. Naming the raw step field and its markers makes the operator
review contract explicit before any backend/WING session or supervised
fader/gain proof.

## Changed

- `.ai/project-layer/wing_mvp_blockers.md` names the raw first operator step
  field in the first blocker next action and current command.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` and `Docs/MVP_WING_SUPERVISED.md`
  require the same markers before supervised proof planning.
- `tools/codex_autoloop/codex_cycle_prompt.md` and
  `tools/codex_autoloop/README.md` mirror the requirement.
- `tests/test_project_director_agent.py` and `tests/test_wing_mvp_rails.py`
  assert the field and markers on selected-action, runbook, MVP, progress,
  prompt, README, and docs-index surfaces.

## Boundary

This report does not grant live authority. It does not contact WING, does not
start a backend/WING session, does not stop or reconfigure local listeners, and
does not close the read-only connection proof blocker.

## Validation

- `PYTHONPATH=backend python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
  passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
  passed: `23 passed in 0.22s`.
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  reported `selected_blocker: WING read-only connection proof`,
  `selected_action` includes `backend_port_prestart_check_operator_sequence_step`,
  `prestart_backend_port_check`, and `unexpected listeners`,
  `guard_status: on_target`, and `guard_stop_the_line: false`.
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
  reported `status: on_target` and `stop_the_line: false`.
- `/opt/homebrew/bin/git diff --check` passed.
- `python3 tools/memory/memory_lint.py` passed.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` passed:
  `3205 passed, 1 skipped, 1 warning in 90.78s`.
