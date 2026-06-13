# Automixer WING MVP Backend Prestart Template Review Fields

Date: 2026-06-13
Status: Validated

## Summary

The shared backend-port prestart failure-detail contract now includes all
generated backend-listener template review fields that the busy-port no-WING
summary already surfaces. Handoff packets, no-WING summaries, proof plans,
Project Director, and runbooks can require one complete fail-closed list before
any real WING read-only session proceeds from a busy backend-port condition.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No blocker closure or readiness update.

## Contract

`backend_port_prestart_check_failure_detail_fields` now also includes:

- `listener_confirmation_template_safe_to_stop_listener_by_gate`
- `listener_confirmation_template_live_authority_granted`
- `listener_confirmation_template_operator_placeholder_fields`
- `listener_confirmation_template_required_edits`

These fields prove that a generated backend-listener confirmation template is
still only an operator-edit aid until the operator replaces placeholders, sets
`template_status: operator_reviewed_and_edited`, and sets
`accepted_by_template: true`.

## Validation

- `/opt/homebrew/bin/python3 -m py_compile backend/wing_readonly_preflight_contract.py tools/no_wing_release_check.py`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py tests/test_codex_direction_guard.py --tb=short`: 235 passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_goal_prestart_template_review_fields_nowing_20260613_r2`: expected fail-closed on existing `127.0.0.1:8765` listener; no hardware contact and `live_write: false`.
- `rg -n "listener_confirmation_template_safe_to_stop_listener_by_gate|listener_confirmation_template_live_authority_granted|listener_confirmation_template_operator_placeholder_fields|listener_confirmation_template_required_edits" /tmp/automixer_goal_prestart_template_review_fields_nowing_20260613_r2/20260613T115543Z_no_wing_release_check/summary.md /tmp/automixer_goal_prestart_template_review_fields_nowing_20260613_r2/20260613T115543Z_no_wing_release_check/summary.json`: confirmed all four fields are present in saved Markdown and JSON summaries.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`: selected `WING read-only connection proof`, guard `on_target`, stop-the-line false, and selected evidence file count 36.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`: `status=on_target`, `stop_the_line=false`, `current_focus=supervised_wing_rack_mvp`.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`: 3221 passed, 1 skipped, 1 warning.
