# Automixer WING MVP Target Defaults Contract

Date: 2026-06-13
Status: Validated

## Summary

The WING read-only preflight target defaults now have one shared contract source:

- `url`: `ws://localhost:8765`
- `channels`: `3`
- `value_types`: `fader,gain,pan`
- `artifact_root`: `artifacts/wing_readonly_preflight`

## Why

The first WING MVP blocker is still `WING read-only connection proof`. Before a
real run, the expected-plan command, capture command, and handoff packet must all
describe the same read-only target. This change prevents stale command text from
passing no-WING validation.

## Changed surfaces

- `backend/wing_readonly_preflight_contract.py` defines the shared target
  defaults.
- `tools/build_wing_readonly_preflight_handoff_packet.py` uses the shared
  defaults for generated handoff packets.
- `tools/no_wing_release_check.py` validates plan, capture, and handoff target
  defaults against the shared contract and includes the result in JSON/Markdown
  summaries.
- Focused tests cover the shared defaults and a stale handoff URL failure.

## Safety boundary

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
  passed: `259 passed in 3.74s`.
- no-WING quick on an unused backend port passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py
  --quick --backend-port 65535 --artifact-root
  /tmp/automixer_goal_target_defaults_contract_nowing_freeport_20260613`.
  The generated summary
  `/tmp/automixer_goal_target_defaults_contract_nowing_freeport_20260613/20260613T155757Z_no_wing_release_check/summary.json`
  reported `status: passed`, `failed_checks: []`, `live_write: false`,
  `connection_policy: no_hardware_no_network_no_live_writes`,
  `expected_target_defaults` matching the shared contract,
  `plan_command_target_defaults_match_contract: true`,
  `capture_command_url: ws://localhost:8765`,
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
- `/opt/homebrew/bin/git diff --check`: passed.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- Full suite passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: `3239 passed, 1 skipped, 1 warning in 166.55s`.
