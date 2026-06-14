# Automixer WING MVP Handoff Declared Director Self-Check

## Summary

The WING read-only handoff packet now declares the Project Director
selected-action self-check fields that no-WING validation recomputes:

- `expected_project_director_selected_action_anchor`
- `project_director_selected_action_anchor_matches_current_directive`
- `expected_project_director_selected_action_source`
- `project_director_selected_action_source_matches_current_directive`
- `actual_project_director_selected_action_sha256`
- `expected_project_director_selected_action_sha256`
- `project_director_selected_action_sha256_available_matches_digest`
- `project_director_selected_action_sha256_matches_anchor`
- `project_director_selected_action_sha256_matches_text`
- `project_director_selected_action_sha256_matches_current_directive`

`tools/no_wing_release_check.py` now fails closed if any declared self-check
field diverges from the current recomputed Project Director directive.

## Boundary

This is offline handoff metadata and release-gate validation. It does not start
the backend, connect to WING, query a console, write to a console, grant live
authority, update readiness, or close `WING read-only connection proof`.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
  (`296 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/build_wing_readonly_preflight_handoff_packet.py tools/no_wing_release_check.py tools/codex_autoloop/update_mvp_progress.py`.
- PASS: Generated handoff packet emitted `project_director_selected_action_source: Current First Command`,
  matching expected/actual Project Director selected-action SHA256
  `b34d0c53a40f512c54bdb59f1a7e555cd9f2f2e8e03fdddee50c2443b23514ed`, all
  Project Director digest match verdicts `true`, and marker count `87`.
- PASS: Direction Guard returned `status: on_target`, score `90`,
  `stop_the_line: false`, and zero drift items.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_handoff_declared_director_self_check_nowing_20260614`
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
  (`3307 passed, 1 skipped, 1 warning`).
