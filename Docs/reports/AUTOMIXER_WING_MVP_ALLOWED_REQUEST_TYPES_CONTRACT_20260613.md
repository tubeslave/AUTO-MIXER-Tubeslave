# Automixer WING MVP Allowed Request Types Contract

Date: 2026-06-13
Status: Validated

## Summary

The WING read-only preflight allowed request types are being moved into one
shared contract source:

- `get_operator_mode`
- `get_state`
- `query_mixer_parameter`

## Why

The first WING MVP blocker is still `WING read-only connection proof`. A real
read-only run must keep its backend request surface narrow and reviewable. The
allowed request list should not drift between runner output and evidence/no-WING
validation.

## Changed surfaces

- `backend/wing_readonly_preflight_contract.py` defines the allowed request
  types.
- `tools/wing_readonly_preflight_runner.py` writes `allowed_request_types` from
  the contract.
- `backend/wing_readonly_preflight_evidence.py` validates sent and expected
  request types against the contract.
- `tools/no_wing_release_check.py` validates generated plans against the
  contract.
- Focused tests cover shared allowed-request binding.

## Safety boundary

- No WING hardware is contacted by this change.
- No backend listener is started, stopped, or reconfigured by this change.
- No OSC/WING writes are authorized by this change.
- No new request types or read targets are added.
- Readiness closure still requires real WING read-only proof and manual review.

## Validation

- Compile check passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  backend/wing_readonly_preflight_contract.py
  backend/wing_readonly_preflight_evidence.py
  tools/wing_readonly_preflight_runner.py
  tools/no_wing_release_check.py
  tests/test_wing_readonly_preflight_runner.py
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_no_wing_release_check.py`.
- Focused runner, evidence, and no-WING tests passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_runner.py
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_no_wing_release_check.py --tb=short`: `321 passed`.
- no-WING quick on an unused backend port passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py
  --quick --backend-port 65535 --artifact-root
  /tmp/automixer_goal_allowed_request_types_contract_20260613`.
  The generated plan
  `/tmp/automixer_goal_allowed_request_types_contract_20260613/20260613T181825Z_no_wing_release_check/wing_readonly_preflight_plan.json`
  reported `allowed_request_types: ["get_operator_mode", "get_state",
  "query_mixer_parameter"]`, `read_only: true`, `live_write: false`,
  `contacts_backend: false`, `queries_console: false`,
  `execution_allowed_by_plan: false`, `live_authority_granted: false`, and the
  expected request sequence for operator mode, state, and `fader`/`gain`/`pan`
  physical readback queries.
- The no-WING handoff packet remained fail-closed:
  `status: passed`, `readiness_closure_allowed_by_packet: false`,
  `readiness_matrix_update_allowed_by_packet: false`,
  `execution_allowed_by_packet: false`, `live_authority_granted: false`,
  `next_manual_artifact_required: real wing_readonly_preflight.v1
  summary.json`.
- WING rails tests passed:
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
- `/opt/homebrew/bin/git diff --check`: passed.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- Full suite passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: `3265 passed, 1 skipped, 1 warning`.
