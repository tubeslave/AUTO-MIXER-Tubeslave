# Automixer WING MVP Handoff Supervised Summary Field Contract

## Summary

The WING read-only handoff packet now declares the supervised proof-plan
summary field contract that no-WING validation recomputes:

- `required_supervised_proof_plan_summary_fields`
- `expected_required_supervised_proof_plan_summary_fields`
- `required_supervised_proof_plan_summary_fields_match_contract`

`tools/no_wing_release_check.py` now fails closed when the packet-declared
expected field list or match flag diverges from the canonical supervised
prestart proof-plan summary field contract.

## Boundary

This is offline handoff metadata and release-gate validation. It does not start
the backend, connect to WING, query a console, write to a console, grant live
authority, update readiness, or close `WING read-only connection proof`.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
  (`297 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/build_wing_readonly_preflight_handoff_packet.py tools/no_wing_release_check.py tools/codex_autoloop/update_mvp_progress.py`.
- PASS: Generated handoff packet emitted identical
  `required_supervised_proof_plan_summary_fields` and
  `expected_required_supervised_proof_plan_summary_fields`, with
  `required_supervised_proof_plan_summary_fields_match_contract: true`.
- PASS: Direction Guard returned `status: on_target`, score `90`, and
  `stop_the_line: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_handoff_supervised_summary_field_contract_nowing_20260614`
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
  (`3308 passed, 1 skipped, 1 warning`).
