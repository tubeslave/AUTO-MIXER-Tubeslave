# Automixer WING MVP Existing Backend Required By Packet Next Action

Date: 2026-06-13

## Summary

The WING MVP rails now require the no-WING handoff summary to show:

- `physical_preflight_requires_existing_backend_connection: true`

before any supervised fader/gain proof planning.

## Why

The handoff packet is an offline operator packet. It can describe the safe
sequence for a later real WING read-only preflight, but that physical preflight
must use the intended already-connected backend/WING session.

## Changed surfaces

- `.ai/project-layer/wing_mvp_blockers.md` requires
  `physical_preflight_requires_existing_backend_connection: true` in the first
  blocker next action and current command.
- `Docs/MVP_WING_SUPERVISED.md` lists missing
  `physical_preflight_requires_existing_backend_connection: true` as a
  stop-the-line condition.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` requires the field in handoff
  validation.
- `tools/codex_autoloop/README.md` and
  `tools/codex_autoloop/codex_cycle_prompt.md` require future cycles to verify
  the field.
- `tests/test_project_director_agent.py` and `tests/test_wing_mvp_rails.py`
  cover the new rail.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md` passed.
- Focused rails validation passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
  returned `23 passed in 0.18s`.
- Project Director selected `WING read-only connection proof`; its selected
  action includes `physical_preflight_requires_existing_backend_connection: true`;
  guard status is `on_target`; `guard_stop_the_line` is `False`.
- Direction guard status is `on_target`; `stop_the_line` is `False`.
- `/opt/homebrew/bin/git diff --check` passed.
- `python3 tools/memory/memory_lint.py` passed.
- Full backend suite passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  returned `3205 passed, 1 skipped, 1 warning in 218.47s (0:03:38)`.
