# Automixer WING MVP Handoff Expected Artifacts Contract

Status: Validated

## Summary

This change makes the WING read-only preflight handoff packet declare the same
expected artifact list as the expected plan. The no-WING release gate now
rejects drift between the packet list, named summary/event globs, and expected
plan artifact list.

## Safety Boundary

- No backend startup.
- No WING transport usage.
- No OSC/WebSocket writes.
- No mixer state mutation.
- No live authority granted.
- No automatic readiness closure.
- The first open blocker remains WING read-only connection proof until real
  hardware evidence is captured and manually reviewed.

## Verification

- Compile check:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  tools/build_wing_readonly_preflight_handoff_packet.py
  tools/no_wing_release_check.py
  tests/test_wing_readonly_preflight_handoff_packet.py
  tests/test_no_wing_release_check.py`: passed.
- Focused handoff/no-WING tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_handoff_packet.py
  tests/test_no_wing_release_check.py --tb=short`: `264 passed`.
- no-WING quick on unused backend port:
  `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/no_wing_release_check.py --quick --backend-port 65535
  --artifact-root
  /tmp/automixer_goal_handoff_expected_artifacts_contract_20260613`: passed.
- Generated no-WING handoff summary:
  `status: passed`, `violations: []`,
  `expected_artifacts_match_plan: true`,
  `summary_glob_matches_plan: true`, and
  `event_log_glob_matches_plan: true`.
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
  --tb=short -q`: `3280 passed, 1 skipped, 1 warning`.

## Next Real-World Step

Run `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` before
the intended backend/WING session, then run
`Docs/runbooks/WING_READONLY_PREFLIGHT.md` against real WING Rack and manually
review the captured artifacts. This offline contract change does not close
readiness.
