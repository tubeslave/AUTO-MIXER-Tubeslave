# Automixer WING MVP Handoff Gate Contract Source - 2026-06-13

Status: Validated

## Scope

This report covers a no-hardware maintenance hardening for the first open WING
MVP blocker, `WING read-only connection proof`.

## Change

The WING read-only handoff gate contracts now have one shared source:

- `backend/wing_readonly_preflight_contract.py`
- `expected_backend_port_prestart_gate()`
- `expected_supervised_prestart_gate()`
- `WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_REQUIRED_MARKERS`
- `WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_MARKERS`

The handoff packet builder and no-WING release-check validator now consume the
same gate contract while preserving the existing public packet fields:
`backend_port_prestart_gate`, `expected_backend_port_prestart_gate`,
`supervised_prestart_gate`, and `expected_supervised_prestart_gate`.

## Safety Boundary

- No WING hardware is contacted by this change.
- No backend listener is started, stopped, or reconfigured by this change.
- No OSC/WING writes are authorized by this change.
- Handoff and no-WING artifacts still grant no live authority.
- Readiness closure still requires real WING read-only proof and manual review.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  backend/wing_readonly_preflight_contract.py
  tools/build_wing_readonly_preflight_handoff_packet.py
  tools/no_wing_release_check.py` passed.
- Focused handoff/no-WING tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_handoff_packet.py
  tests/test_no_wing_release_check.py tests/test_project_director_agent.py
  tests/test_codex_direction_guard.py tests/test_wing_mvp_rails.py
  --tb=short` passed: `248 passed in 3.11s`.
- no-WING quick on an unused backend port passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py
  --quick --backend-port 65535 --artifact-root
  /tmp/automixer_goal_handoff_gate_contract_source_nowing_freeport_20260613`.
  The generated handoff artifact reported `status: passed`,
  `operator_sequence_count: 10`, `selected_evidence_file_count: 44`,
  `plan_artifact_violations: []`, `handoff_violations: []`, and
  `readiness_closure_allowed_by_packet: false`.
- Project Director:
  `selected_blocker=WING read-only connection proof`,
  `selected_blocker_status=tooling_ready_waits_real_wing`,
  `guard_status=on_target`, `selected_evidence_file_count=44`.
- Direction guard:
  `status=on_target`, `stop_the_line=false`,
  `current_focus=supervised_wing_rack_mvp`.
- Full suite passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: `3225 passed, 1 skipped, 1 warning in 166.64s`.
