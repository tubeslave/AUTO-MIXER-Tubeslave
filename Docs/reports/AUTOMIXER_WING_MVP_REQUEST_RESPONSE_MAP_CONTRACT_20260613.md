# Automixer WING MVP Request Response Map Contract

Date: 2026-06-13
Status: Validated

## Summary

The WING read-only preflight request-to-response mapping is being moved into one
shared contract source:

- `get_operator_mode` -> `operator_mode_status`
- `get_state` -> `state_update`
- `query_mixer_parameter` -> `mixer_parameter_value` or
  `mixer_parameter_read_blocked`

## Why

The first WING MVP blocker is still `WING read-only connection proof`. A real
read-only run must fail closed when a valid-looking response type appears in the
wrong request context. Runner runtime handling and evidence event-log validation
should use the same mapping.

## Changed surfaces

- `backend/wing_readonly_preflight_contract.py` defines the request-to-response
  mapping.
- `tools/wing_readonly_preflight_runner.py` derives runtime expected response
  sets from the mapping.
- `backend/wing_readonly_preflight_evidence.py` validates event-log response
  context against the mapping.
- Focused tests cover shared mapping binding.

## Safety boundary

- No WING hardware is contacted by this change.
- No backend listener is started, stopped, or reconfigured by this change.
- No OSC/WING writes are authorized by this change.
- No new request types, response types, or read targets are added.
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
  tests/test_no_wing_release_check.py --tb=short`: `323 passed`.
- no-WING quick on an unused backend port passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py
  --quick --backend-port 65535 --artifact-root
  /tmp/automixer_goal_request_response_map_contract_20260613`.
  The generated plan
  `/tmp/automixer_goal_request_response_map_contract_20260613/20260613T184222Z_no_wing_release_check/wing_readonly_preflight_plan.json`
  reported `read_only: true`, `live_write: false`, `contacts_backend: false`,
  `queries_console: false`, `execution_allowed_by_plan: false`,
  `live_authority_granted: false`, and the expected request sequence for
  operator mode, state, and `fader`/`gain`/`pan` physical readback queries.
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
  --tb=short -q`: `3267 passed, 1 skipped, 1 warning`.
