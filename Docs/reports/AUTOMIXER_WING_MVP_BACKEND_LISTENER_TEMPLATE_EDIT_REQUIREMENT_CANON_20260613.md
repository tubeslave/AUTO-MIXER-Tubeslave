# Automixer WING MVP Backend Listener Template Edit Requirement Canon

Date: 2026-06-13
Status: Validated

## Summary

The current WING read-only blocker now carries the same backend-listener
template edit requirement as the no-WING validator. Future Project Director
output must preserve the stricter contract: generated-template confirmations
cannot clear the backend-port prestart gate until operator placeholders are
replaced, `template_status: operator_reviewed_and_edited` is set, and
`accepted_by_template: true` is set.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No blocker closure or readiness update.

## Updated Canon

- `.ai/project-layer/wing_mvp_blockers.md` cites the template edit gate report
  in first-blocker evidence.
- The first-blocker next action now requires
  `listener_confirmation_template_required_edits` to include the reviewed
  status and accepted-by-template edits.
- `Docs/reports/MVP_PROGRESS_LATEST.md` is regenerated from the blocker table.
- Project Director and rail tests assert the stricter fields.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_mvp_rails.py tests/test_project_director_agent.py tests/test_codex_direction_guard.py --tb=short`: 23 passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`: selected `WING read-only connection proof`, guard `on_target`, selected evidence file count 32, and selected action includes `listener_confirmation_template_required_edits`, `template_status: operator_reviewed_and_edited`, and `accepted_by_template: true`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`: `status=on_target`, `stop_the_line=false`, `current_focus=supervised_wing_rack_mvp`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_mvp_rails.py tests/test_project_director_agent.py tests/test_codex_direction_guard.py tests/test_no_wing_release_check.py --tb=short`: 235 passed.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_goal_template_edit_canon_nowing_20260613`: expected fail-closed on existing `127.0.0.1:8765` listener; summary exposes `listener_confirmation_template_required_edits` with reviewed-template and accepted-by-template edits.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`: 3221 passed, 1 skipped, 1 warning.
