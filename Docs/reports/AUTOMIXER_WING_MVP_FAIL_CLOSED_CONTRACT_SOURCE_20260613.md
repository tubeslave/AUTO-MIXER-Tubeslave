# Automixer WING MVP Fail-Closed Contract Source - 2026-06-13

Status: Validated

## Scope

This report covers a no-hardware maintenance hardening for the first open WING
MVP blocker, `WING read-only connection proof`.

## Change

The WING readonly expected-plan fail-closed condition list now has a single
canonical source:

- `backend/wing_readonly_preflight_contract.py`
- `WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS`

The runner, evidence intake, and no-WING release-check validator now consume that
shared tuple while preserving their existing public aliases.

## Safety Boundary

- No WING hardware is contacted by this change.
- No backend listener is started, stopped, or reconfigured by this change.
- No OSC/WING writes are authorized by this change.
- Readiness closure still requires real WING read-only proof and manual review.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  backend/wing_readonly_preflight_contract.py
  backend/wing_readonly_preflight_evidence.py
  tools/wing_readonly_preflight_runner.py tools/no_wing_release_check.py`
  passed.
- Focused shared-contract consumers:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_runner.py
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_no_wing_release_check.py tests/test_project_director_agent.py
  tests/test_codex_direction_guard.py --tb=short` passed:
  `302 passed in 3.91s`.
- no-WING quick on an unused backend port passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py
  --quick --backend-port 65535 --artifact-root
  /tmp/automixer_goal_fail_closed_contract_source_nowing_freeport_20260613`.
  The generated WING readonly plan artifact reported `status: passed`,
  `plan_artifact_violations: []`, and the shared fail-closed list.
- Project Director:
  `selected_blocker=WING read-only connection proof`,
  `selected_blocker_status=tooling_ready_waits_real_wing`,
  `guard_status=on_target`, `guard_stop_the_line=false`,
  `selected_evidence_file_count=43`.
- Direction guard:
  `status=on_target`, `stop_the_line=false`,
  `current_focus=supervised_wing_rack_mvp`.
- Full suite passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: `3224 passed, 1 skipped, 1 warning in 166.28s`.
