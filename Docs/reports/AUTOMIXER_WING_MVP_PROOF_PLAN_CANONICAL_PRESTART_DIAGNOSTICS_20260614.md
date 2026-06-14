# Automixer WING MVP Proof Plan Canonical Prestart Diagnostics

## Summary

The supervised WING proof plan now carries canonical expected
accepted-candidate prestart diagnostics:

- `readonly_preflight_review_decision_expected_candidate_prestart_source_ids`
- `readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source`

`tools/no_wing_release_check.py` parses the reported plan fields, recomputes
the expected fields through the shared
`backend.wing_readonly_preflight_contract` helper, fails closed when the
reported fields diverge, and renders the recomputed expected fields in both
`summary.json` and `summary.md`.

## Boundary

This is an offline report and release-gate diagnostics change. It does not
start the backend, connect to WING, query a console, write to a console, grant
live authority, or close readiness. The first blocker still requires real WING
read-only artifacts and manual review.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile backend/supervised_wing_proof_plan.py tools/no_wing_release_check.py`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py --tb=short -x`
  (`264 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py --tb=short -x`
  (`338 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
  regenerated latest progress with `Generated: 2026-06-14T05:28:54Z`,
  `Closed blockers: 0`, and first open blocker
  `WING read-only connection proof`.
- PASS: Project Director selected `WING read-only connection proof`, status
  `tooling_ready_waits_real_wing`, and returned `guard_status: on_target`,
  `guard_score: 90`, `guard_stop_the_line: false`.
- PASS: Direction Guard returned `status: on_target`, score `90`, and
  `stop_the_line: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_goal_proof_plan_canonical_prestart_diagnostics_nowing_20260614`
  failed closed on the existing `127.0.0.1:8765` listener with
  `backend_port_listener_requires_operator_confirmation`,
  `listener_confirmation_template_status: pending_operator_edit`,
  `backend_port_prestart_gate_live_authority_granted: false`, and
  `backend_port_prestart_gate_readiness_closure_allowed: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  (`3304 passed, 1 skipped, 1 warning`).
