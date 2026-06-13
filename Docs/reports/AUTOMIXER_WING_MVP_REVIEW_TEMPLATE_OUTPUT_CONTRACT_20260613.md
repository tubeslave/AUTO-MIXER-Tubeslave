# Automixer WING MVP Review Template Output Contract - 2026-06-13

Status: Validated

## Scope

This report covers a no-hardware safety-contract hardening for the first open
WING MVP blocker, `WING read-only connection proof`.

## Change

The WING read-only review decision template output path now has one shared
source:

- `backend/wing_readonly_preflight_contract.py`
- `WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT`
- `WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT`

The read-only preflight runner, handoff packet builder, review artifact tool,
and no-WING release-check validator now consume the shared path contract while
preserving the existing public path:

`artifacts/wing_readonly_preflight_review_decisions/pending_review_decision_template.json`

## Safety Boundary

- No WING hardware is contacted by this change.
- No backend listener is started, stopped, or reconfigured by this change.
- No OSC/WING writes are authorized by this change.
- Review, handoff, and no-WING artifacts still grant no live authority.
- Readiness closure still requires real WING read-only proof and manual review.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  backend/wing_readonly_preflight_contract.py
  tools/review_wing_readonly_preflight_artifacts.py
  tools/no_wing_release_check.py
  tools/build_wing_readonly_preflight_handoff_packet.py
  tools/wing_readonly_preflight_runner.py` passed.
- Focused WING readonly runner, handoff, review, and no-WING tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_review_tool.py
  tests/test_wing_readonly_preflight_handoff_packet.py
  tests/test_wing_readonly_preflight_runner.py
  tests/test_no_wing_release_check.py --tb=short`
  passed: `269 passed in 3.72s`.
- Focused review, no-WING, director, guard, and rails tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_review_tool.py
  tests/test_wing_readonly_preflight_handoff_packet.py
  tests/test_wing_readonly_preflight_runner.py
  tests/test_no_wing_release_check.py tests/test_project_director_agent.py
  tests/test_codex_direction_guard.py tests/test_wing_mvp_rails.py --tb=short`
  passed: `292 passed in 3.06s`.
- no-WING quick on an unused backend port passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py
  --quick --backend-port 65535 --artifact-root
  /tmp/automixer_goal_review_template_output_contract_nowing_freeport_20260613`.
  The generated summary reported `status: passed`, `failed_checks: []`,
  `live_write: false`,
  `connection_policy: no_hardware_no_network_no_live_writes`,
  `review_command` containing `--decision-template-output` followed by
  `artifacts/wing_readonly_preflight_review_decisions/pending_review_decision_template.json`,
  `review_command_writes_decision_template: true`, and
  `readiness_closure_allowed_by_packet: false`.
- Project Director:
  `selected_blocker=WING read-only connection proof`,
  `selected_blocker_status=tooling_ready_waits_real_wing`,
  `guard_status=on_target`, `guard_score=90`,
  `selected_evidence_file_count=44`.
- Direction guard:
  `status=on_target`, `stop_the_line=false`,
  `current_focus=supervised_wing_rack_mvp`.
- `git diff --check` passed.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py` passed.
- Full suite passed:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: `3233 passed, 1 skipped, 1 warning in 172.38s`.
