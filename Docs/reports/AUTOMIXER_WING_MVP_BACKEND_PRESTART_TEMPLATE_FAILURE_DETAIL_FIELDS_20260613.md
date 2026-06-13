# Automixer WING MVP Backend Prestart Template Failure Detail Fields

Date: 2026-06-13
Status: Validated

## Summary

The shared backend-port prestart failure-detail contract now includes the
generated backend-listener template status and authority-denial fields. Handoff
packets, no-WING summaries, proof plans, Project Director, and runbooks now name
the same template evidence before any WING read-only session can proceed from a
busy backend-port condition.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No blocker closure or readiness update.

## Contract

`backend_port_prestart_check_failure_detail_fields` now includes:

- `listener_confirmation_template_status`
- `listener_confirmation_template_accepted_by_template`
- `listener_confirmation_template_execution_allowed_by_template`
- `listener_confirmation_template_readiness_closure_allowed_by_template`

## Validation

- `/opt/homebrew/bin/python3 -m py_compile backend/wing_readonly_preflight_contract.py tools/no_wing_release_check.py`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py tests/test_codex_direction_guard.py --tb=short`: 235 passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_goal_prestart_template_failure_fields_nowing_20260613`: expected fail-closed on existing `127.0.0.1:8765` listener; `backend_port_free` detail exposed `listener_confirmation_template_status: pending_operator_edit`, `listener_confirmation_template_accepted_by_template: false`, `listener_confirmation_template_execution_allowed_by_template: false`, and `listener_confirmation_template_readiness_closure_allowed_by_template: false`.
- `rg -n "listener_confirmation_template_status|listener_confirmation_template_accepted_by_template|listener_confirmation_template_execution_allowed_by_template|listener_confirmation_template_readiness_closure_allowed_by_template|backend_port_prestart_check_failure_detail_fields" /tmp/automixer_goal_prestart_template_failure_fields_nowing_20260613/20260613T114236Z_no_wing_release_check/summary.md /tmp/automixer_goal_prestart_template_failure_fields_nowing_20260613/20260613T114236Z_no_wing_release_check/summary.json`: confirmed the template fields are present in saved JSON and Markdown summaries.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`: selected `WING read-only connection proof`, guard `on_target`, stop-the-line false, and selected evidence file count 35.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`: `status=on_target`, `stop_the_line=false`, `current_focus=supervised_wing_rack_mvp`.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`: 3221 passed, 1 skipped, 1 warning.
