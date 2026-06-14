# Automixer WING MVP Current Command Director Action Digest

## Summary

`## Current First Command` now requires the no-WING handoff summary to show the
Project Director selected-action digest proof:

- `project_director_selected_action_anchor`
- `project_director_selected_action_anchor_matches_current_directive: true`
- `project_director_selected_action_source: Current First Command`
- `project_director_selected_action_source_matches_current_directive: true`
- `project_director_selected_action_sha256`
- `project_director_selected_action_sha256_available: true`
- `project_director_selected_action_sha256_matches_anchor: true`
- `project_director_selected_action_sha256_matches_text: true`
- `project_director_selected_action_sha256_matches_current_directive: true`
- `project_director_current_first_command_baseline_coverage_required: true`
- `project_director_current_first_command_baseline_coverage_passed: true`
- `project_director_current_first_command_baseline_missing_markers: []`

The same markers are now part of `CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS`,
so Project Director keeps using `Current First Command` only while those fields
remain present.

## Boundary

This is offline Project Director/autoloop contract sync. It does not start the
backend, connect to WING, query a console, write to a console, grant live
authority, update readiness, or close `WING read-only connection proof`.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_cycle_prompt.py tests/test_codex_autoloop_contract.py --tb=short -x`
  (`25 passed`).
- PASS: Project Director selected `WING read-only connection proof`, status
  `tooling_ready_waits_real_wing`, `selected_action_source: Current First Command`,
  `selected_action_sha256: fb3c94759f636f764fc442808e75fc1e521caf317e81a370f2e3ffad56e0be12`,
  `current_first_command_baseline_coverage_marker_count: 81`,
  `current_first_command_baseline_coverage_passed: true`, and
  `current_first_command_baseline_missing_markers: []`.
- PASS: Direction Guard returned `status: on_target`, score `90`,
  `stop_the_line: false`, and zero drift items.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_current_command_director_action_digest_nowing_20260614`
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
  (`3306 passed, 1 skipped, 1 warning`).
