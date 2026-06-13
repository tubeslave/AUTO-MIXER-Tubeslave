# Automixer WING MVP Backend Prestart Must Pass By Packet Next Action

Date: 2026-06-13
Status: Validated

## Summary

The WING MVP rails now require the no-WING handoff summary to show:

- `backend_port_prestart_check_must_pass_before_starting_backend: true`

before any supervised fader/gain proof planning.

## Why

The handoff packet is offline operator guidance for a later read-only proof
path. Its backend-port prestart listener check must pass before starting the
intended backend/WING session.

## Changed surfaces

- `.ai/project-layer/wing_mvp_blockers.md` requires
  `backend_port_prestart_check_must_pass_before_starting_backend: true` in the
  first blocker next action and current command.
- `Docs/MVP_WING_SUPERVISED.md` lists missing
  `backend_port_prestart_check_must_pass_before_starting_backend: true` as a
  stop-the-line condition.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` requires the field in handoff
  validation.
- `tools/codex_autoloop/README.md` and
  `tools/codex_autoloop/codex_cycle_prompt.md` require future cycles to verify
  the field.
- `tests/test_project_director_agent.py` and `tests/test_wing_mvp_rails.py`
  cover the new rail.

## Validation

- Regenerated `Docs/reports/MVP_PROGRESS_LATEST.md` with
  `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/codex_autoloop/update_mvp_progress.py --blockers
  .ai/project-layer/wing_mvp_blockers.md --output
  Docs/reports/MVP_PROGRESS_LATEST.md`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
  tests/test_codex_direction_guard.py`: 23 passed in 0.23s.
- Project Director JSON check selected `WING read-only connection proof`, found
  `backend_port_prestart_check_must_pass_before_starting_backend: true` in the
  selected action, and reported `guard_status: on_target` with
  `guard_stop_the_line: false`.
- Direction guard JSON check reported `status: on_target` and
  `stop_the_line: false`.
- `/opt/homebrew/bin/git diff --check`: passed.
- `python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3205 passed, 1 skipped, 1 warning in 218.52s (0:03:38).
