# Automixer WING MVP Current Command Director Digest Rendered Fields

## Summary

`Current First Command` baseline coverage now requires the full rendered
Project Director selected-action digest field set from the no-WING handoff
summary:

- `expected_project_director_selected_action_anchor`
- `expected_project_director_selected_action_source: Current First Command`
- `actual_project_director_selected_action_sha256`
- `expected_project_director_selected_action_sha256`
- `project_director_selected_action_sha256_available_matches_digest: true`
- `project_director_current_first_command_baseline_coverage_marker_count`

Those fields are now checked in the baseline marker tuple, autoloop prompt,
autoloop README, Project Director tests, and WING progress rail tests.

## Boundary

This is offline Project Director/autoloop contract sync. It does not start the
backend, connect to WING, query a console, write to a console, grant live
authority, update readiness, or close `WING read-only connection proof`.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_cycle_prompt.py tests/test_codex_autoloop_contract.py --tb=short -x`
  (`25 passed`).
- PASS: Project Director selected `WING read-only connection proof`, status
  `tooling_ready_waits_real_wing`, `selected_action_source: Current First Command`,
  `selected_action_sha256: b34d0c53a40f512c54bdb59f1a7e555cd9f2f2e8e03fdddee50c2443b23514ed`,
  `current_first_command_baseline_coverage_marker_count: 87`,
  `current_first_command_baseline_coverage_passed: true`, and
  `current_first_command_baseline_missing_markers: []`.
- PASS: Direction Guard returned `status: on_target`, score `90`,
  `stop_the_line: false`, and zero drift items.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_current_command_director_digest_rendered_fields_nowing_20260614`
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
