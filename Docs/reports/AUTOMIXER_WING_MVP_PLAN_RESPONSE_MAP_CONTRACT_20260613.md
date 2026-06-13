# Automixer WING MVP Plan Response Map Contract - 2026-06-13

Status: Validated

## Scope

Expose the shared read-only WING preflight request-to-response mapping in the
offline expected plan and validate the same mapping in evidence/no-WING checks.

## Safety Boundary

This is offline/no-WING contract hardening. It does not contact WING hardware,
does not close readiness, and does not grant live authority.

## Validation

- Compile check:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  backend/wing_readonly_preflight_evidence.py
  tools/wing_readonly_preflight_runner.py
  tools/no_wing_release_check.py
  tests/test_wing_readonly_preflight_runner.py
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_no_wing_release_check.py
  tests/test_wing_readonly_preflight_review_tool.py
  tests/test_wing_readonly_preflight_review_decision.py
  tests/test_wing_readonly_preflight_review_decision_tool.py`: passed.
- Focused WING preflight/no-WING/review tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_runner.py
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_no_wing_release_check.py
  tests/test_wing_readonly_preflight_review_tool.py
  tests/test_wing_readonly_preflight_review_decision.py
  tests/test_wing_readonly_preflight_review_decision_tool.py --tb=short`:
  `363 passed`.
- no-WING quick on unused backend port:
  `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/no_wing_release_check.py --quick --backend-port 65535
  --artifact-root
  /tmp/automixer_goal_plan_response_map_contract_20260613`: passed.
- Generated no-WING preflight plan:
  `status: passed`, `violations: []`, `failed_checks: []`, and
  `expected_response_types_by_request_type` exactly:
  `get_operator_mode -> operator_mode_status`,
  `get_state -> state_update`,
  `query_mixer_parameter -> mixer_parameter_value |
  mixer_parameter_read_blocked`.
- WING rails tests:
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
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- `/opt/homebrew/bin/git diff --check`: passed.
- Full suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: `3268 passed, 1 skipped, 1 warning`.

## Next real-world step

Run `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` before
the intended backend/WING session, then run
`Docs/runbooks/WING_READONLY_PREFLIGHT.md` against real WING Rack and manually
review the captured artifacts. This offline contract change does not close
readiness.
