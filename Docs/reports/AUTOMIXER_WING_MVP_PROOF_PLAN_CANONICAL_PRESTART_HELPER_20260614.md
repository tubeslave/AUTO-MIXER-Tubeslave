# Automixer WING MVP Proof Plan Canonical Prestart Helper

## Summary

The no-WING `next_wing_proof_plan` gate now validates accepted-candidate
prestart traces through the shared
`backend.wing_readonly_preflight_contract` helper. The gate no longer relies on
a shorter local check that only inspected the backend prestart command fields.

The valid proof-report fixture now carries the full accepted-candidate trace:

- expected summary/events artifact paths
- summary/events path binding booleans
- backend prestart command fields
- backend prestart summary path and SHA
- prestart summary type/status/connection policy
- no-live/no-backend/no-console/no-test booleans
- backend prestart check status

## Boundary

This is an offline release-gate validation change. It does not start the
backend, connect to WING, query a console, write to a console, grant live
authority, or close readiness. The first blocker still requires real WING
read-only artifacts and manual review.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py --tb=short -x`
  (`263 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py --tb=short -x`
  (`337 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
  regenerated latest progress with `Generated: 2026-06-14T05:16:39Z`,
  `Closed blockers: 0`, and first open blocker
  `WING read-only connection proof`.
- PASS: Project Director selected `WING read-only connection proof`, status
  `tooling_ready_waits_real_wing`, and returned `guard_status: on_target`.
- PASS: Direction Guard returned `on_target`, score `90`, and
  `stop_the_line: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_goal_proof_plan_canonical_prestart_helper_nowing_20260614`
  failed closed on the existing `127.0.0.1:8765` listener with
  `backend_port_listener_requires_operator_confirmation`,
  `listener_confirmation_template_status: pending_operator_edit`,
  `backend_port_prestart_gate_live_authority_granted: false`, and
  `backend_port_prestart_gate_readiness_closure_allowed: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  (`3303 passed, 1 skipped, 1 warning`).
