# Automixer WING MVP Handoff Proof-Plan Summary Fields

## Summary

The WING read-only handoff packet now carries
`required_supervised_proof_plan_summary_fields`, a shared-contract list of the
proof-plan summary fields the operator must have before moving from read-only
proof into supervised proof:

- `readonly_preflight_review_decision_candidate_prestart_contract_valid`
- `readonly_preflight_review_decision_candidate_prestart_source_ids`
- `readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source`
- `readonly_preflight_review_decision_expected_candidate_prestart_source_ids`
- `readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source`

The no-WING handoff packet validator now fails closed if that list is missing
or stale, and the no-WING summary renders both the declared and expected field
lists.

## Boundary

This is an offline handoff and release-gate validation change. It does not
start the backend, connect to WING, query a console, write to a console, grant
live authority, or close readiness. The first blocker still requires real WING
read-only artifacts and manual review.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile backend/wing_readonly_preflight_contract.py tools/build_wing_readonly_preflight_handoff_packet.py tools/no_wing_release_check.py`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py --tb=short -x`
  (`283 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py --tb=short -x`
  (`357 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
  regenerated latest progress with `Generated: 2026-06-14T05:44:06Z`,
  `Closed blockers: 0`, and first open blocker
  `WING read-only connection proof`.
- PASS: Project Director selected `WING read-only connection proof`, status
  `tooling_ready_waits_real_wing`, and returned `guard_status: on_target`,
  `guard_score: 90`, `guard_stop_the_line: false`.
- PASS: Direction Guard returned `status: on_target`, score `90`, and
  `stop_the_line: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_goal_handoff_proof_plan_summary_fields_nowing_20260614`
  failed closed on the existing `127.0.0.1:8765` listener with
  `backend_port_listener_requires_operator_confirmation`,
  `listener_confirmation_template_status: pending_operator_edit`,
  `backend_port_prestart_gate_live_authority_granted: false`, and
  `backend_port_prestart_gate_readiness_closure_allowed: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  (`3305 passed, 1 skipped, 1 warning`).
