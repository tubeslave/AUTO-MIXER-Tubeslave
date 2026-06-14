# Automixer WING MVP Handoff Director Action Digest

## Summary

The WING read-only handoff packet now carries a Project Director selected-action
anchor:

- `project_director_selected_action_anchor`
- `project_director_selected_action_source`
- `project_director_selected_action_sha256`
- `project_director_selected_action_sha256_available`
- `project_director_current_first_command_baseline_coverage_required`
- `project_director_current_first_command_baseline_coverage_passed`
- `project_director_current_first_command_baseline_coverage_marker_count`
- `project_director_current_first_command_baseline_missing_markers`

The no-WING schema gate validates that the anchor matches the current selected
action, that the SHA256 matches the anchor text, that the top-level digest
matches the current directive, and that the baseline coverage fields match the
current-first-command contract.

## Boundary

This is offline handoff metadata. It does not start the backend, connect to
WING, query a console, write to a console, grant live authority, update
readiness, or close `WING read-only connection proof`. Real WING read-only
artifacts and manual review remain required.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py --tb=short -x`
  (`288 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/build_wing_readonly_preflight_handoff_packet.py --format json --output /tmp/wing_handoff_director_action_digest_packet_20260614.json --quiet`
  emitted `project_director_selected_action_source: Current First Command`,
  `project_director_selected_action_sha256: b7e89bc1f6d61e172b5536418d9b1cbffce3d3665c3a89f9e36686d48629b322`,
  `project_director_selected_action_sha256_available: true`,
  `project_director_current_first_command_baseline_coverage_required: true`,
  `project_director_current_first_command_baseline_coverage_passed: true`, and
  `project_director_current_first_command_baseline_missing_markers: []`.
- PASS: Project Director selected `WING read-only connection proof`, status
  `tooling_ready_waits_real_wing`, `selected_action_source: Current First Command`,
  `selected_action_sha256: b7e89bc1f6d61e172b5536418d9b1cbffce3d3665c3a89f9e36686d48629b322`,
  `current_first_command_baseline_coverage_required: true`,
  `current_first_command_baseline_coverage_passed: true`,
  `current_first_command_baseline_missing_markers: []`, and marker count `69`.
- PASS: Direction Guard returned `status: on_target`, score `90`,
  `stop_the_line: false`, and zero drift items.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_handoff_director_action_digest_nowing_20260614`
  failed closed on the existing `127.0.0.1:8765` listener with
  `backend_port_listener_requires_operator_confirmation`,
  `listener_confirmation_template_status: pending_operator_edit`,
  `listener_confirmation_template_accepted_by_template: false`,
  `listener_confirmation_template_execution_allowed_by_template: false`,
  `backend_port_prestart_gate_live_authority_granted: false`, and
  `backend_port_prestart_gate_readiness_closure_allowed: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py tools/build_wing_readonly_preflight_handoff_packet.py tools/codex_autoloop/update_mvp_progress.py`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  (`3306 passed, 1 skipped, 1 warning`).
