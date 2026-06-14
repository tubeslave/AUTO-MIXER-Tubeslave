# Automixer WING MVP Proof Plan Preflight Command Bindings

## Summary

The supervised WING proof plan now binds its fallback
`wing_readonly_preflight_required` command to the same fail-closed inputs used
by the handoff packet and runbook:

- `--prestart-summary`
- `--expected-plan`
- `readonly_preflight_command_prestart_summary`
- `readonly_preflight_command_expected_plan`

`tools/no_wing_release_check.py` now validates and summarizes these fields in
the `next_wing_proof_plan` release gate output.

## Boundary

This is an offline proof-plan/release-gate change. It does not start the
backend, connect to WING, query a console, write to a console, grant live
authority, or close readiness. The first blocker still requires real WING
read-only artifacts and manual review.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py --tb=short -x`
  (`274 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile backend/supervised_wing_proof_plan.py tools/no_wing_release_check.py`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py --tb=short -x`
  (`336 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
  regenerated latest progress with `Generated: 2026-06-14T04:48:15Z`.
- PASS: Project Director selected `WING read-only connection proof`, used
  `Current First Command + blocker table baseline`, reported
  `guard_status: on_target`, score `90`, and `stop_broad_work: true`.
- PASS: Direction guard returned `on_target`, score `90`,
  `stop_the_line: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_goal_proof_plan_preflight_command_bindings_nowing_20260614`
  failed closed on the existing `127.0.0.1:8765` listener with
  `backend_port_listener_requires_operator_confirmation`,
  `listener_confirmation_template_status: pending_operator_edit`,
  `backend_port_prestart_gate_live_authority_granted: false`, and
  `backend_port_prestart_gate_readiness_closure_allowed: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  (`3302 passed, 1 skipped, 1 warning`).
