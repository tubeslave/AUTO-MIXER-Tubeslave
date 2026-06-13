# Automixer WING MVP Plan Output Contract - 2026-06-13

Status: Validated

## Scope

This report covers a no-hardware safety-contract hardening for the first open
WING MVP blocker, `WING read-only connection proof`.

## Change

The WING read-only preflight expected plan output path now has one shared
contract source:

- `backend/wing_readonly_preflight_contract.py`
- `WING_READONLY_PREFLIGHT_PLAN_OUTPUT`

The handoff builder and no-WING release-check validator now consume the shared
contract value while preserving the existing public path:

`artifacts/wing_readonly_preflight_plan/latest_plan.json`

The no-WING evidence snapshot now reports and validates:

- `plan_command_output_matches_contract`
- `capture_command_expected_plan_matches_contract`
- `handoff_packet_plan_output_matches_contract`
- `handoff_packet_plan_output_matches_plan_command`

If the plan, capture, or handoff commands drift to a stale expected-plan path,
no-WING validation fails closed before any real backend/WING session.

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
- Focused handoff, no-WING, director, guard, and rails tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_handoff_packet.py
  tests/test_no_wing_release_check.py tests/test_project_director_agent.py
  tests/test_codex_direction_guard.py tests/test_wing_mvp_rails.py --tb=short`
  passed: `258 passed in 3.70s`.
- no-WING quick on an unused backend port passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py
  --quick --backend-port 65535 --artifact-root
  /tmp/automixer_goal_plan_output_contract_nowing_freeport_20260613`.
  The generated summary
  `/tmp/automixer_goal_plan_output_contract_nowing_freeport_20260613/20260613T154636Z_no_wing_release_check/summary.json`
  reported `status: passed`, `failed_checks: []`, `live_write: false`,
  `connection_policy: no_hardware_no_network_no_live_writes`,
  `plan_command_output` set to
  `artifacts/wing_readonly_preflight_plan/latest_plan.json`,
  `plan_command_output_matches_contract: true`,
  `capture_command_expected_plan` set to
  `artifacts/wing_readonly_preflight_plan/latest_plan.json`,
  `capture_command_expected_plan_matches_contract: true`,
  `handoff_packet_plan_output` set to
  `artifacts/wing_readonly_preflight_plan/latest_plan.json`,
  `handoff_packet_plan_output_matches_contract: true`,
  `handoff_packet_plan_output_matches_plan_command: true`, and
  `readiness_closure_allowed_by_evidence: false`.
- Project Director:
  `current_mvp=supervised_wing_rack_proof`,
  `selected_blocker=WING read-only connection proof`,
  `selected_blocker_status=tooling_ready_waits_real_wing`,
  `guard_status=on_target`, `guard_score=90`,
  `guard_stop_the_line=false`,
  `selected_evidence_file_count=44`,
  `continue_goal_mode=true`.
- Direction guard:
  `status=on_target`, `stop_the_line=false`,
  `current_focus=supervised_wing_rack_mvp`, `score=90`.
- Full suite passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: `3238 passed, 1 skipped, 1 warning in 204.85s`.
