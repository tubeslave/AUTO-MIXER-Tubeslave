# Automixer WING MVP Director Action Coverage Digest

## Summary

Project Director now publishes machine-readable proof that its shorter
`Current First Command` selected action still covers the WING safety/proof
baseline:

- `selected_action_sha256`
- `current_first_command_baseline_coverage_required`
- `current_first_command_baseline_coverage_passed`
- `current_first_command_baseline_coverage_marker_count`
- `current_first_command_baseline_missing_markers`

`current_first_command_baseline_missing_markers` is empty for the current first
blocker, and the selected action SHA256 binds the exact text that downstream
Goal-mode handoff should audit.

## Boundary

This is offline Project Director/autoloop metadata. It does not start the
backend, connect to WING, query a console, write to a console, grant live
authority, close readiness, or close the first blocker. Real WING read-only
artifacts and manual review remain required.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
  (`11 passed`).
- PASS: Project Director selected `WING read-only connection proof`, status
  `tooling_ready_waits_real_wing`, `selected_action_source: Current First Command`,
  `current_first_command_baseline_coverage_required: true`,
  `current_first_command_baseline_coverage_passed: true`,
  `current_first_command_baseline_missing_markers: []`,
  `current_first_command_baseline_coverage_marker_count: 69`, and
  `selected_action_sha256: b7e89bc1f6d61e172b5536418d9b1cbffce3d3665c3a89f9e36686d48629b322`.
- PASS: Direction Guard returned `status: on_target`, score `90`,
  `stop_the_line: false`, and zero drift items.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_goal_director_action_coverage_digest_nowing_20260614`
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
