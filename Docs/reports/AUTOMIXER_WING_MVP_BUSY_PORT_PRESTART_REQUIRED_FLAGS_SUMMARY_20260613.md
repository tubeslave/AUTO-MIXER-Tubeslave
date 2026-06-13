# Automixer WING MVP Busy-Port Prestart Required Flags Summary

Date: 2026-06-13
Status: Validated

## Summary

Failed busy-port no-WING summaries now surface
`backend_port_prestart_check_required: true` and
`backend_port_prestart_check_must_pass_before_starting_backend: true` in both
`summary.json` and `summary.md`. The first fail-closed prestart artifact now
states the same mandatory backend-port gate boundary that handoff packets,
proof plans, Project Director, and runbooks require before a real WING
read-only session can proceed.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No blocker closure or readiness update.

## Contract

When `backend_port_free` fails because an existing listener is detected, the
saved result detail includes:

- `backend_port_prestart_check_required: true`
- `backend_port_prestart_check_must_pass_before_starting_backend: true`

## Validation

- `/opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py backend/wing_readonly_preflight_contract.py`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py tests/test_codex_direction_guard.py --tb=short`: 235 passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_goal_busy_port_prestart_required_flags_summary_nowing_20260613`: expected fail-closed on existing `127.0.0.1:8765` listener; exit status 1, no hardware contact, and `live_write: false`.
- `rg -n "backend_port_prestart_check_required|backend_port_prestart_check_must_pass_before_starting_backend|live_write|backend_port_free|listener_confirmation_template_status" /tmp/automixer_goal_busy_port_prestart_required_flags_summary_nowing_20260613/20260613T123551Z_no_wing_release_check/summary.md /tmp/automixer_goal_busy_port_prestart_required_flags_summary_nowing_20260613/20260613T123551Z_no_wing_release_check/summary.json`: confirmed both flags are present in saved Markdown and JSON summaries.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`: selected `WING read-only connection proof`, guard `on_target`, stop-the-line false, and selected evidence file count 39.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`: `status=on_target`, `stop_the_line=false`, `current_focus=supervised_wing_rack_mvp`.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`: 3221 passed, 1 skipped, 1 warning.
