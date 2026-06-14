# Automixer WING MVP Current First Command Proof-Plan Summary Fields

## Summary

`## Current First Command` now carries the same supervised proof-plan summary
field contract that the handoff packet, no-WING gate, blocker row, and autoloop
prompt already require:

- `required_supervised_proof_plan_summary_fields`
- `expected_required_supervised_proof_plan_summary_fields`
- `required_supervised_proof_plan_summary_fields_match_contract: true`
- `readonly_preflight_review_decision_expected_candidate_prestart_source_ids`
- `readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source`

This makes the Project Director primary selected-action source self-contained
instead of depending on the blocker-row baseline to recover these fields.

## Boundary

This is an offline planning/contract sync. It does not start the backend,
connect to WING, query a console, write to a console, grant live authority, or
close readiness. The first blocker still requires real WING read-only artifacts
and manual review.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
  (`11 passed`).
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
  regenerated latest progress with `Generated: 2026-06-14T06:04:42Z`,
  `Closed blockers: 0`, and first open blocker
  `WING read-only connection proof`.
- PASS: Project Director selected `WING read-only connection proof`, status
  `tooling_ready_waits_real_wing`, `selected_action_source: Current First Command + blocker table baseline`,
  `guard_status: on_target`, `guard_score: 90`,
  `guard_stop_the_line: false`, and selected action includes
  `required_supervised_proof_plan_summary_fields`.
- PASS: Direction Guard returned `status: on_target`, score `90`, and
  `stop_the_line: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_goal_current_first_command_proof_plan_summary_fields_nowing_20260614`
  failed closed on the existing `127.0.0.1:8765` listener with
  `backend_port_listener_requires_operator_confirmation`,
  `listener_confirmation_template_status: pending_operator_edit`,
  `backend_port_prestart_gate_live_authority_granted: false`, and
  `backend_port_prestart_gate_readiness_closure_allowed: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  (`3305 passed, 1 skipped, 1 warning`).
