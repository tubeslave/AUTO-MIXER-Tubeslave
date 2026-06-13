# Automixer WING MVP Handoff Request Sequence Contract - 2026-06-13

Status: Validated

## Scope

Expose the read-only WING preflight request sequence in the operator handoff
packet and validate that it matches the generated expected plan artifact and
expected no-WING sequence.

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
  tests/test_no_wing_release_check.py --tb=short`: `261 passed`.
- no-WING quick on unused backend port:
  `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/no_wing_release_check.py --quick --backend-port 65535
  --artifact-root
  /tmp/automixer_goal_handoff_request_sequence_contract_20260613`: passed.
- Generated no-WING handoff summary:
  `status: passed`, `violations: []`, `failed_checks: []`,
  `request_count: 5`, `request_sequence_matches_plan: true`,
  `request_sequence_matches_expected: true`, request types exactly
  `get_operator_mode`, `get_state`, and three `query_mixer_parameter`
  requests, with addresses `/ch/3/fdr`, `/ch/3/in/set/trim`, and `/ch/3/pan`.
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
  --tb=short -q`: `3277 passed, 1 skipped, 1 warning`.

## Next real-world step

Run `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` before
the intended backend/WING session, then run
`Docs/runbooks/WING_READONLY_PREFLIGHT.md` against real WING Rack and manually
review the captured artifacts. This offline contract change does not close
readiness.
