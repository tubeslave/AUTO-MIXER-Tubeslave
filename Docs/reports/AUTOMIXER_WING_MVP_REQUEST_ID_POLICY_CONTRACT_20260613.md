# Automixer WING MVP Request ID Policy Contract

Date: 2026-06-13
Status: Validated

## Summary

The WING read-only preflight request-id policy is being moved into one shared
contract source:

`required_for_every_query_mixer_parameter_response`

## Why

The first WING MVP blocker is still `WING read-only connection proof`. A real
read-only run must prove that each `mixer_parameter_value` or
`mixer_parameter_read_blocked` event corresponds to the expected
`query_mixer_parameter` request. The policy should not drift between runner
output and evidence/no-WING validation.

## Changed surfaces

- `backend/wing_readonly_preflight_contract.py` defines the request-id policy.
- `tools/wing_readonly_preflight_runner.py` writes the policy from the contract.
- `backend/wing_readonly_preflight_evidence.py` validates against the contract.
- `tools/no_wing_release_check.py` validates generated plans against the
  contract.
- Focused tests cover shared request-id policy binding.

## Safety boundary

- No WING hardware is contacted by this change.
- No backend listener is started, stopped, or reconfigured by this change.
- No OSC/WING writes are authorized by this change.
- Request-id matching for physical readbacks remains mandatory.
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
  tests/test_no_wing_release_check.py --tb=short`: `319 passed`.
- no-WING quick on an unused backend port passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py
  --quick --backend-port 65535 --artifact-root
  /tmp/automixer_goal_request_id_policy_contract_20260613`.
  The generated plan
  `/tmp/automixer_goal_request_id_policy_contract_20260613/20260613T180746Z_no_wing_release_check/wing_readonly_preflight_plan.json`
  reported `request_id_policy:
  required_for_every_query_mixer_parameter_response`, `read_only: true`,
  `live_write: false`, `contacts_backend: false`, `queries_console: false`,
  `execution_allowed_by_plan: false`, `live_authority_granted: false`, and a
  request sequence with request IDs for `fader`, `gain`, and `pan`.
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
  --tb=short -q`: `3263 passed, 1 skipped, 1 warning`.
