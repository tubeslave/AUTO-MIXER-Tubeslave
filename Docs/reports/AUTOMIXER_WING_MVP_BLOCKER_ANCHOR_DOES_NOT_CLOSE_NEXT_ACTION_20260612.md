# Automixer WING MVP Blocker Anchor Does Not Close Next Action

Date: 2026-06-12

## Summary

The first open WING MVP blocker now explicitly requires the no-WING handoff
summary to show `mvp_blocker_anchor.packet_does_not_close_blocker: true` before
any supervised proof planning.

## Why

Offline handoff packets can prepare the operator review flow, but they must not
be mistaken for real WING proof or blocker closure. Project Director uses the
blocker table as its selected action source, so the blocker text must carry that
boundary directly.

## Changed

- `.ai/project-layer/wing_mvp_blockers.md` names
  `mvp_blocker_anchor.packet_does_not_close_blocker: true` in the first blocker
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
  - selected action includes `mvp_blocker_anchor.packet_does_not_close_blocker: true`
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
