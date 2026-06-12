# Automixer WING MVP Operator Required By Packet Next Action

Date: 2026-06-13

## Summary

The WING MVP rails now require the no-WING handoff summary to show:

- `operator_required: true`

before any supervised fader/gain proof planning.

## Why

The handoff packet is an offline operator packet. It can describe the safe
sequence for a later real WING read-only preflight, but it must not be treated
as autonomous execution permission.

## Changed surfaces

- `.ai/project-layer/wing_mvp_blockers.md` requires `operator_required: true` in
  the first blocker next action and current command.
- `Docs/MVP_WING_SUPERVISED.md` lists missing `operator_required: true` as a
  stop-the-line condition.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` requires the field in handoff
  validation.
- `tools/codex_autoloop/README.md` and
  `tools/codex_autoloop/codex_cycle_prompt.md` require future cycles to verify
  the field.
- `tests/test_project_director_agent.py` and `tests/test_wing_mvp_rails.py`
  cover the new rail.

## Validation

- `PYTHONPATH=backend python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
  passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
  passed: `23 passed in 0.24s`.
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  reported `selected_blocker: WING read-only connection proof`,
  `selected_action` includes `operator_required: true`,
  `guard_status: on_target`, and `guard_stop_the_line: false`.
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
  reported `status: on_target` and `stop_the_line: false`.
- `/opt/homebrew/bin/git diff --check` passed.
- `python3 tools/memory/memory_lint.py` passed.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` passed:
  `3205 passed, 1 skipped, 1 warning in 176.51s`.
