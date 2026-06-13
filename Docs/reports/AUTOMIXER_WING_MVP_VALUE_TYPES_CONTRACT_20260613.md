# Automixer WING MVP Value Types Contract

Date: 2026-06-13
Status: Validated

## Summary

The WING read-only preflight supported value types now have one shared contract
source:

- `fader`
- `gain`
- `pan`

The default command value-types string is derived from that contract as
`fader,gain,pan`.

## Why

The first WING MVP blocker is still `WING read-only connection proof`. The real
read-only preflight must stay bounded to the current proof scope; runner and
handoff validators should not silently drift from the generated command
contract.

## Changed surfaces

- `backend/wing_readonly_preflight_contract.py` defines supported read-only
  value types.
- `tools/wing_readonly_preflight_runner.py` uses the shared allow-list for
  `--value-types` validation.
- `tools/build_wing_readonly_preflight_handoff_packet.py` uses the shared
  allow-list for handoff validation.
- `tools/no_wing_release_check.py` introspects runner-supported value types and
  fails closed if they diverge from the contract.
- Focused tests cover shared allow-lists, summary fields, and stale runner
  supported-value rejection.

## Safety boundary

- No WING hardware is contacted by this change.
- No backend listener is started, stopped, or reconfigured by this change.
- No OSC/WING writes are authorized by this change.
- The read-only proof scope remains `fader`, `gain`, and `pan`.
- Readiness closure still requires real WING read-only proof and manual review.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  backend/wing_readonly_preflight_contract.py
  tools/wing_readonly_preflight_runner.py
  tools/build_wing_readonly_preflight_handoff_packet.py
  tools/no_wing_release_check.py` passed.
- Focused runner, handoff, and no-WING tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_runner.py
  tests/test_wing_readonly_preflight_handoff_packet.py
  tests/test_no_wing_release_check.py --tb=short`
  passed: `268 passed in 3.79s`.
- no-WING quick on an unused backend port passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py
  --quick --backend-port 65535 --artifact-root
  /tmp/automixer_goal_value_types_contract_nowing_freeport_20260613`.
  The generated summary
  `/tmp/automixer_goal_value_types_contract_nowing_freeport_20260613/20260613T162310Z_no_wing_release_check/summary.json`
  reported `status: passed`, `failed_checks: []`, `live_write: false`,
  `connection_policy: no_hardware_no_network_no_live_writes`,
  `expected_supported_value_types: ["fader", "gain", "pan"]`,
  `runner_supported_value_types: ["fader", "gain", "pan"]`,
  `runner_supported_value_types_match_contract: true`,
  `runner_argparse_target_defaults_match_contract: true`,
  `plan_command_target_defaults_match_contract: true`,
  `capture_command_target_defaults_match_contract: true`,
  `handoff_packet_target_defaults_match_contract: true`, and
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
- Expanded focused runner, handoff, no-WING, director, guard, and rails tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_runner.py
  tests/test_wing_readonly_preflight_handoff_packet.py
  tests/test_no_wing_release_check.py tests/test_project_director_agent.py
  tests/test_codex_direction_guard.py tests/test_wing_mvp_rails.py --tb=short`
  passed: `291 passed in 3.07s`.
- `/opt/homebrew/bin/git diff --check`: passed.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- Full suite passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: `3242 passed, 1 skipped, 1 warning in 168.87s`.
