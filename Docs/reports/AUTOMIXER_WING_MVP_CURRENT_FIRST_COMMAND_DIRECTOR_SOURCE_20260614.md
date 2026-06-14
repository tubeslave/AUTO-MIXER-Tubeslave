# Automixer WING MVP Current First Command Director Source

## Summary

Project Director and the generated MVP progress report now prefer the
`## Current First Command` section from `.ai/project-layer/wing_mvp_blockers.md`
for selected action text. When that section differs from the table row, the
first open blocker table row is appended as the baseline action so older safety
constraints remain visible. The table row also remains the source for selected
blocker status and evidence files.

## Boundary

This is an offline Goal-mode routing change only. It does not start the backend,
connect to WING, query a console, write to a console, or grant live authority.
The first blocker still requires real WING read-only artifacts and manual
review before readiness can close.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short`
  (`11 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --output .paperclip/reports/project_director_current_first_command_director_source_20260614.json`.
- PASS: Project Director selected `WING read-only connection proof`, source
  `Current First Command + blocker table baseline`, and preserved accepted
  prestart diagnostics plus baseline table action.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json --output .paperclip/reports/codex_direction_guard_current_first_command_director_source_20260614.json`
  returned `on_target`, score `90`, `stop_the_line: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/codex_autoloop/update_mvp_progress.py tools/codex_autoloop/project_director_agent.py`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  (`3299 passed, 1 skipped, 1 warning`).
