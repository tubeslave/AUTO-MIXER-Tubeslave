# Automixer WING MVP Decision-Ready Status Contract

Date: 2026-06-13

## Status

Validated.

## Boundary

This change is offline and read-only. It does not contact WING, does not mutate
console state, does not grant live authority, and does not close readiness rows.

## Scope

- Bind the decision-review CLI's accepted-ready status to the shared supervised
  prestart gate contract.
- Verify the CLI and `expected_supervised_prestart_gate()` require the same
  status before supervised proof planning.

## Validation

- Compile check:
  `/opt/homebrew/bin/python3 -m py_compile
  tools/review_wing_readonly_preflight_decisions.py
  tests/test_wing_readonly_preflight_review_decision_tool.py`: passed.
- Focused decision/proof-plan tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_review_decision_tool.py
  tests/test_wing_readonly_preflight_review_decision.py
  tests/test_supervised_wing_proof_plan.py --tb=short`: `40 passed`.
- no-WING quick on unused backend port:
  `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/no_wing_release_check.py --quick --backend-port 65535
  --artifact-root
  /tmp/automixer_goal_decision_ready_status_contract_20260613`: passed.
- no-WING summary:
  `type: no_wing_release_check`, `status: passed`, `failed_checks: []`,
  `connection_policy: no_hardware_no_network_no_live_writes`,
  `live_write: false`.
- backend-port prestart summary:
  `type: no_wing_release_check`, `status: passed`,
  `connection_policy: no_hardware_no_network_no_live_writes`,
  `live_write: false`, `contacts_backend: false`,
  `queries_console: false`, `executes_tests: false`,
  backend-port result `backend_port_free: passed`.
- no-WING handoff:
  `operator_sequence_count: 10`,
  `readiness_closure_allowed_by_packet: false`,
  `live_authority_granted: false`,
  `supervised_prestart_gate.requires_operator_decision_status:
  accepted_decision_ready`.
- decision-review ledger overlay:
  `review_exit_codes.accepted_decision_ready: 0`.
- WING rails tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_project_director_agent.py tests/test_codex_direction_guard.py
  tests/test_wing_mvp_rails.py --tb=short`: `23 passed`.
- Direction guard:
  `status: on_target`, `current_focus: supervised_wing_rack_mvp`,
  `stop_the_line: false`, `score: 90`.
- Project Director:
  `current_mvp: supervised_wing_rack_proof`,
  `selected_blocker: WING read-only connection proof`,
  `selected_blocker_status: tooling_ready_waits_real_wing`,
  `guard_status: on_target`, `guard_stop_the_line: false`.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- Full suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: `3250 passed, 1 skipped, 1 warning`.

## Next real-world step

Run `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` before
the intended backend/WING session, then run `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
against real WING Rack and manually review the captured artifacts.
