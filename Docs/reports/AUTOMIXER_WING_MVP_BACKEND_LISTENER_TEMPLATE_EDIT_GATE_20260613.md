# Automixer WING MVP Backend Listener Template Edit Gate

Date: 2026-06-13
Status: Validated

## Summary

The backend-listener confirmation validator now rejects partially edited
generated templates. A JSON derived from
`backend-listener-confirmation-template.json` must replace operator
placeholders, mark the template as operator-reviewed, and explicitly set
`accepted_by_template: true` before it can clear the local backend-port
prestart gate.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from template-derived confirmation.

## Contract

Accepted confirmation JSON still requires matching host, port, PID, and command
line. If it carries generated-template metadata, it must also show:

- `template_status: operator_reviewed_and_edited`
- `accepted_by_template: true`
- no `REPLACE_WITH_*` placeholder values in `operator_reviewed_at` or
  `operator_name`

Partial edits keep the gate failed with `listener_confirmation_status:
invalid`.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py tests/test_codex_direction_guard.py --tb=short`: 235 passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_goal_listener_template_edit_gate_nowing_20260613`: failed closed on existing `127.0.0.1:8765` listener and generated a pending template with the new required edits.
- Generated template inspection confirmed `template_status: pending_operator_edit`, `accepted_by_template: false`, placeholder operator fields, and required edits for `operator_reviewed_and_edited` plus `accepted_by_template: true`.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`: 3221 passed, 1 skipped, 1 warning.
