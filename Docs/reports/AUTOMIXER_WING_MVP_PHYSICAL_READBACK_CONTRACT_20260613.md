# Automixer WING MVP Physical Readback Contract

Date: 2026-06-13
Status: Validated

## Summary

The WING read-only preflight physical readback identity now has one shared
contract source:

- statuses: `verified_from_physical_readback`
- sources: `physical_readback`, `wing_physical_readback`, `physical_wing`,
  `wing`

## Why

The first WING MVP blocker is still `WING read-only connection proof`. The
runner and evidence intake must agree on what counts as real physical readback
before any artifact can become a manual readiness-update candidate.

## Changed surfaces

- `backend/wing_readonly_preflight_contract.py` defines physical readback
  statuses and sources.
- `tools/wing_readonly_preflight_runner.py` uses the shared physical readback
  identity.
- `backend/wing_readonly_preflight_evidence.py` uses the same identity when
  accepting or rejecting captured artifacts.
- `tools/no_wing_release_check.py` compares runner and evidence identities to
  the contract and fails closed on drift.
- Focused tests cover shared identities, summary fields, and stale runner or
  evidence proof semantics.

## Safety boundary

- No WING hardware is contacted by this change.
- No backend listener is started, stopped, or reconfigured by this change.
- No OSC/WING writes are authorized by this change.
- The change only hardens what offline tooling accepts as physical proof.
- Readiness closure still requires real WING read-only proof and manual review.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  backend/wing_readonly_preflight_contract.py
  tools/wing_readonly_preflight_runner.py
  backend/wing_readonly_preflight_evidence.py
  tools/no_wing_release_check.py` passed.
- Focused runner, evidence, and no-WING tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_runner.py
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_no_wing_release_check.py --tb=short`
  passed: `303 passed in 3.79s`.
- no-WING quick on an unused backend port passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py
  --quick --backend-port 65535 --artifact-root
  /tmp/automixer_goal_physical_readback_contract_nowing_freeport_20260613`.
  The generated summary
  `/tmp/automixer_goal_physical_readback_contract_nowing_freeport_20260613/20260613T163457Z_no_wing_release_check/summary.json`
  reported `status: passed`, `failed_checks: []`, `live_write: false`,
  `connection_policy: no_hardware_no_network_no_live_writes`,
  `runner_physical_readback_statuses_match_contract: true`,
  `runner_physical_readback_sources_match_contract: true`,
  `evidence_physical_readback_statuses_match_contract: true`,
  `evidence_physical_readback_sources_match_contract: true`, and
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
- Expanded focused runner, evidence, no-WING, director, guard, and rails tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_runner.py
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_no_wing_release_check.py tests/test_project_director_agent.py
  tests/test_codex_direction_guard.py tests/test_wing_mvp_rails.py --tb=short`
  passed: `326 passed in 3.08s`.
- `/opt/homebrew/bin/git diff --check`: passed.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- Full suite passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: `3245 passed, 1 skipped, 1 warning in 177.03s`.
