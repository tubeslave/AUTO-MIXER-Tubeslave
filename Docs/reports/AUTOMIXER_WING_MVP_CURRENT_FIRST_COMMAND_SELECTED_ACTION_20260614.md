# Automixer WING MVP Current First Command Selected Action

## Summary

Project Director now uses `Current First Command` directly as the next required
action only when that command covers an explicit safety/proof marker contract.
If any marker is missing, the progress tool keeps the older
`Current First Command + blocker table baseline` fallback.

The current command was also tightened with the listener-template denial fields
that were previously only recovered from the blocker-row baseline:

- `listener_confirmation_template_accepted_by_template: false`
- `listener_confirmation_template_operator_placeholder_fields`
- `listener_confirmation_template_execution_allowed_by_template: false`
- `listener_confirmation_template_readiness_closure_allowed_by_template: false`

This keeps Goal-mode action text focused without dropping fail-closed listener
handling, supervised-prestart gating, proof-plan summary fields, or
accepted-candidate prestart-trace blockers.

## Boundary

This is an offline Project Director/progress-report selection change. It does
not start the backend, connect to WING, query a console, write to a console,
grant live authority, close readiness, or close the first blocker. Real WING
read-only artifacts and manual review remain required.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
  (`11 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
  regenerated latest progress with `Generated: 2026-06-14T06:14:01Z`,
  `Closed blockers: 0`, first open blocker
  `WING read-only connection proof`, and
  `Next required action source: Current First Command`.
- PASS: Project Director selected `WING read-only connection proof`, status
  `tooling_ready_waits_real_wing`, `selected_action_source: Current First Command`,
  `guard_status: on_target`, `guard_score: 90`, and
  `guard_stop_the_line: false`.
- PASS: Direction Guard returned `status: on_target`, score `90`,
  `stop_the_line: false`, and zero drift items.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_goal_current_first_command_selected_action_nowing_20260614`
  failed closed on the existing `127.0.0.1:8765` listener with
  `backend_port_listener_requires_operator_confirmation`,
  `listener_confirmation_template_status: pending_operator_edit`,
  `listener_confirmation_template_accepted_by_template: false`,
  `listener_confirmation_template_execution_allowed_by_template: false`,
  `backend_port_prestart_gate_live_authority_granted: false`, and
  `backend_port_prestart_gate_readiness_closure_allowed: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  (`3305 passed, 1 skipped, 1 warning`).
