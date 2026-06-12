# Automixer WING MVP Blocker Supervised No-Live Authority Next Action

Date: 2026-06-12

## Summary

The first open WING MVP blocker now explicitly requires
`supervised_prestart_gate.grants_live_authority: false` in the no-WING handoff
summary before any supervised proof planning.

## Why

The handoff packet and runbook already enforce that the supervised prestart gate
does not grant live authority. Project Director uses the blocker table as its
selected action source, so the blocker text must carry the same explicit
boundary.

## Changed

- `.ai/project-layer/wing_mvp_blockers.md` names
  `supervised_prestart_gate.grants_live_authority: false` in the first blocker
  next action and current command.
- `Docs/reports/MVP_PROGRESS_LATEST.md` was regenerated from the blocker table.
- `tests/test_project_director_agent.py` and `tests/test_wing_mvp_rails.py`
  assert that the selected action and blocker table expose the field.

## Boundary

This report does not grant live authority. It does not contact WING, does not
start a backend/WING session, and does not close the read-only connection proof
blocker.

## Validation

- `PYTHONPATH=backend python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
  - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
  - `23 passed`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  - `selected_blocker: WING read-only connection proof`
  - selected action includes `supervised_prestart_gate.grants_live_authority: false`
  - `guard_status: on_target`
  - `guard_stop_the_line: false`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
  - `status: on_target`
  - `stop_the_line: false`
- `/opt/homebrew/bin/git diff --check`
  - PASS
- `python3 tools/memory/memory_lint.py`
  - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  - `3205 passed, 1 skipped, 1 warning`
