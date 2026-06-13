# Automixer WING MVP Backend Listener Confirmation Template

Date: 2026-06-13
Status: Validated

## Summary

The no-WING backend-port prestart gate now writes a pending
`backend-listener-confirmation-template.json` when it fails closed on an
existing local listener. The template copies the current listener PID and
command-line diagnostics so the operator can edit a safer starting point instead
of hand-copying the entire confirmation artifact.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from the generated template.

## Contract

The generated template stays pending and invalid until operator edit:

- `status: operator_review_required`
- `confirms_intended_backend: false`
- `safe_to_stop_listener_by_gate: false`
- `live_authority_granted: false`
- `accepted_by_template: false`
- `execution_allowed_by_template: false`
- `readiness_closure_allowed_by_template: false`

The no-WING summary exposes:

- `listener_confirmation_template_path`
- `listener_confirmation_template_status: pending_operator_edit`
- `listener_confirmation_template_safe_to_stop_listener_by_gate: false`
- `listener_confirmation_template_live_authority_granted: false`

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py tests/test_codex_direction_guard.py --tb=short`: 233 passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_goal_listener_template_nowing_20260613`: failed closed on existing `127.0.0.1:8765` listener and wrote `/tmp/automixer_goal_listener_template_nowing_20260613/20260613T103341Z_no_wing_release_check/backend-listener-confirmation-template.json`.
- Generated template inspection confirmed `status: operator_review_required`, `confirms_intended_backend: false`, `safe_to_stop_listener_by_gate: false`, and `live_authority_granted: false`.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`: 3219 passed, 1 skipped, 1 warning.
