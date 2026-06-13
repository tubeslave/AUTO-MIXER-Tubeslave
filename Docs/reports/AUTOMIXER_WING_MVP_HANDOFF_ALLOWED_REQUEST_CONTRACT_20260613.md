# Automixer WING MVP Handoff Allowed Request Contract - 2026-06-13

Status: Validated

## Scope

Expose the shared read-only WING preflight allowed request types and request-id
policy in the operator handoff packet and validate that both match the shared
contract and generated expected plan artifact.

## Safety Boundary

This is offline/no-WING contract hardening. It does not contact WING hardware,
does not close readiness, and does not grant live authority.

## Validation

- Compile check:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  tools/build_wing_readonly_preflight_handoff_packet.py
  tools/no_wing_release_check.py
  tests/test_wing_readonly_preflight_handoff_packet.py
  tests/test_no_wing_release_check.py`: passed.
- Focused handoff/no-WING tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_handoff_packet.py
  tests/test_no_wing_release_check.py --tb=short`: `257 passed`.
- no-WING quick on unused backend port:
  `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/no_wing_release_check.py --quick --backend-port 65535
  --artifact-root
  /tmp/automixer_goal_handoff_allowed_request_contract_20260613`: passed.
- Generated no-WING handoff summary:
  `status: passed`, `violations: []`, `failed_checks: []`,
  `allowed_request_types_match_plan: true`,
  `request_id_policy_matches_plan: true`,
  `allowed_request_types: ["get_operator_mode", "get_state",
  "query_mixer_parameter"]`, and
  `request_id_policy: required_for_every_query_mixer_parameter_response`.
- WING rails tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_project_director_agent.py tests/test_codex_direction_guard.py
  tests/test_wing_mvp_rails.py --tb=short`: `23 passed`.
- Direction guard:
  `status: on_target`, `current_focus: supervised_wing_rack_mvp`,
  `stop_the_line: false`, `score: 90`.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- `/opt/homebrew/bin/git diff --check`: passed.
- Full suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: `3273 passed, 1 skipped, 1 warning`.

## Next real-world step

Run `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` before
the intended backend/WING session, then run
`Docs/runbooks/WING_READONLY_PREFLIGHT.md` against real WING Rack and manually
review the captured artifacts. This offline contract change does not close
readiness.
