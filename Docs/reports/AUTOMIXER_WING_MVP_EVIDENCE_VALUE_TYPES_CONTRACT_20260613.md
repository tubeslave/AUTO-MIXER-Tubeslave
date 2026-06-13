# Automixer WING MVP Evidence Value-Types Contract

Date: 2026-06-13

## Status

Validated.

## Boundary

This change is offline and read-only. It does not contact WING, does not mutate
console state, does not grant live authority, and does not close readiness rows.

## Scope

- Bind WING read-only preflight evidence `ALLOWED_VALUE_TYPES` to the shared
  runner/handoff value-type contract.
- Make no-WING validation report and fail closed on evidence value-type drift.

## Validation

- Compile check:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  backend/wing_readonly_preflight_evidence.py
  tools/no_wing_release_check.py
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_no_wing_release_check.py`: passed.
- Focused evidence/no-WING tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_no_wing_release_check.py --tb=short`: `279 passed`.
- no-WING quick on unused backend port:
  `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/no_wing_release_check.py --quick --backend-port 65535
  --artifact-root
  /tmp/automixer_goal_evidence_value_types_contract_20260613`: passed.
- no-WING summary:
  `type: no_wing_release_check`, `status: passed`, `failed_checks: []`,
  `connection_policy: no_hardware_no_network_no_live_writes`,
  `live_write: false`.
- WING preflight value-type contract fields:
  `expected_supported_value_types: ["fader", "gain", "pan"]`,
  `runner_supported_value_types: ["fader", "gain", "pan"]`,
  `evidence_allowed_value_types: ["fader", "gain", "pan"]`,
  `runner_supported_value_types_match_contract: true`,
  `evidence_allowed_value_types_match_contract: true`.
- backend-port prestart summary:
  `type: no_wing_release_check`, `status: passed`,
  `connection_policy: no_hardware_no_network_no_live_writes`,
  `live_write: false`, `contacts_backend: false`,
  `queries_console: false`, `executes_tests: false`,
  backend-port result `backend_port_free: passed`.
- no-WING handoff:
  `operator_sequence_count: 10`,
  `readiness_closure_allowed_by_packet: false`,
  `live_authority_granted: false`.
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
- `git diff --check`: passed.
- Full suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: `3253 passed, 1 skipped, 1 warning`.

## Next real-world step

Run `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` before
the intended backend/WING session, then run `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
against real WING Rack and manually review the captured artifacts.
