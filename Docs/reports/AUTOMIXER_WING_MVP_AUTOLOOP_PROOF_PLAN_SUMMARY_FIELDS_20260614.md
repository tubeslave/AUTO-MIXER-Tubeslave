# Automixer WING MVP Autoloop Proof-Plan Summary Fields

## Summary

The Codex autoloop prompt and README now require the same handoff proof-plan
summary field contract that the no-WING gate validates before any supervised
fader/gain proof:

- `required_supervised_proof_plan_summary_fields`
- `expected_required_supervised_proof_plan_summary_fields`
- `required_supervised_proof_plan_summary_fields_match_contract: true`
- `readonly_preflight_review_decision_expected_candidate_prestart_source_ids`
- `readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source`

This keeps autonomous Goal-mode cycles aligned with the handoff packet and
prevents future cycles from checking only older supervised prestart markers.

## Boundary

This is an offline autoloop instruction and static-test sync. It does not start
the backend, connect to WING, query a console, write to a console, grant live
authority, or close readiness. The first blocker still requires real WING
read-only artifacts and manual review.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_cycle_prompt.py tests/test_codex_autoloop_contract.py --tb=short -x`
  (`25 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
  regenerated latest progress with `Generated: 2026-06-14T05:55:16Z`,
  `Closed blockers: 0`, and first open blocker
  `WING read-only connection proof`.
- PASS: Project Director selected `WING read-only connection proof`, status
  `tooling_ready_waits_real_wing`, and returned `guard_status: on_target`,
  `guard_score: 90`, `guard_stop_the_line: false`.
- PASS: Direction Guard returned `status: on_target`, score `90`, and
  `stop_the_line: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_goal_autoloop_proof_plan_summary_fields_nowing_20260614`
  failed closed on the existing `127.0.0.1:8765` listener with
  `backend_port_listener_requires_operator_confirmation`,
  `listener_confirmation_template_status: pending_operator_edit`,
  `backend_port_prestart_gate_live_authority_granted: false`, and
  `backend_port_prestart_gate_readiness_closure_allowed: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  (`3305 passed, 1 skipped, 1 warning`).
