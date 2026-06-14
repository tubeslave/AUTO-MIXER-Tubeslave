# Automixer WING MVP Proof Plan Operator Confirmation Bindings

## Summary

The supervised WING proof plan fallback `wing_readonly_preflight_required`
command now uses the canonical read-only preflight contract:

- `--channels 3`
- `--operator-confirm-channels 3`
- `--operator-confirm-session-ready`
- `--value-types fader,gain,pan`
- `--prestart-summary artifacts/no_wing_release_check/<run_id>/summary.json`
- `--expected-plan artifacts/wing_readonly_preflight_plan/latest_plan.json`

`tools/no_wing_release_check.py` now validates and summarizes these bindings in
the `next_wing_proof_plan` gate output.

## Boundary

This is an offline proof-plan/release-gate change. It does not start the
backend, connect to WING, query a console, write to a console, grant live
authority, or close readiness. The first blocker still requires real WING
read-only artifacts and manual review.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile backend/supervised_wing_proof_plan.py tools/no_wing_release_check.py`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py --tb=short -x`
  (`274 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py --tb=short -x`
  (`336 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
  regenerated latest progress with `Generated: 2026-06-14T05:05:46Z`,
  `Closed blockers: 0`, and first open blocker
  `WING read-only connection proof`.
- PASS: Project Director selected `WING read-only connection proof`, status
  `tooling_ready_waits_real_wing`, and returned `guard_status: on_target`.
- PASS: Direction Guard returned `on_target`, score `90`, and
  `stop_the_line: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_goal_proof_plan_operator_confirmation_bindings_nowing_20260614`
  failed closed on the existing `127.0.0.1:8765` listener with
  `backend_port_listener_requires_operator_confirmation`,
  `listener_confirmation_template_status: pending_operator_edit`,
  `backend_port_prestart_gate_live_authority_granted: false`, and
  `backend_port_prestart_gate_readiness_closure_allowed: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  (`3302 passed, 1 skipped, 1 warning`).
