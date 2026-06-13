# Automixer WING MVP Backend Listener Pending Summary Fields Canon

Date: 2026-06-13
Status: Validated

## Summary

The current WING read-only blocker now requires the pending backend-listener
template summary fields emitted by the no-WING gate. Project Director and
`MVP_PROGRESS_LATEST.md` must show
`listener_confirmation_template_accepted_by_template: false` and
`listener_confirmation_template_operator_placeholder_fields` before any real
WING preflight proceeds from a busy backend-port condition.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No blocker closure or readiness update.

## Updated Canon

- `.ai/project-layer/wing_mvp_blockers.md` cites the pending summary fields
  report in first-blocker evidence.
- The first-blocker next action requires the new pending-template summary
  fields.
- `Docs/reports/MVP_PROGRESS_LATEST.md` is regenerated from the blocker table.
- Project Director and rail tests assert the fields.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_mvp_rails.py tests/test_project_director_agent.py tests/test_codex_direction_guard.py --tb=short`: 23 passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`: selected `WING read-only connection proof`, guard `on_target`, evidence file count 33, and selected action includes `listener_confirmation_template_accepted_by_template: false` plus `listener_confirmation_template_operator_placeholder_fields`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py tests/test_codex_direction_guard.py --tb=short`: 235 passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`: `status=on_target`, `stop_the_line=false`, `current_focus=supervised_wing_rack_mvp`.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`: 3221 passed, 1 skipped, 1 warning.
