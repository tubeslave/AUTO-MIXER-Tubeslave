# Automixer WING MVP Selected Blocker Contract - 2026-06-13

Status: Validated

## Scope

Bind the WING read-only preflight selected MVP blocker name and status to
shared contract constants.

## Safety Boundary

This is offline/no-WING contract hardening. It does not contact WING hardware,
does not close readiness, and does not grant live authority.

## Validation

- Compile check:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  backend/wing_readonly_preflight_contract.py
  backend/wing_readonly_preflight_evidence.py
  tools/wing_readonly_preflight_runner.py
  tools/no_wing_release_check.py
  tests/test_wing_readonly_preflight_runner.py
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_no_wing_release_check.py`: passed.
- Focused WING preflight/no-WING tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_runner.py
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_no_wing_release_check.py --tb=short`: `317 passed`.
- no-WING quick on unused backend port:
  `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/no_wing_release_check.py --quick --backend-port 65535
  --artifact-root
  /tmp/automixer_goal_selected_blocker_contract_20260613`: passed.
- no-WING summary:
  `selected_blocker: WING read-only connection proof`,
  `selected_blocker_status: tooling_ready_waits_real_wing`,
  `readiness_closure_allowed_by_packet: false`,
  `live_authority_granted: false`.
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
- `git diff --check`: passed.
- Full suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: `3261 passed, 1 skipped, 1 warning`.

## Next real-world step

Run `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` before
the intended backend/WING session, then run
`Docs/runbooks/WING_READONLY_PREFLIGHT.md` against real WING Rack and manually
review the captured artifacts. This offline contract change does not close
readiness.
