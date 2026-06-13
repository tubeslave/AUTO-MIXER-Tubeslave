# WING MVP Supervised Gate Expected-Artifact Marker

## Summary

The final WING supervised prestart gate now explicitly requires
`expected-artifact-bound accepted source id` before pointing operators to
`WING_SUPERVISED_FADER_GAIN_PROOF`.

The required marker list is now:

- `accepted_decision_ready`
- `valid accepted-candidate prestart trace`
- `expected-artifact-bound accepted source id`
- `WING_SUPERVISED_FADER_GAIN_PROOF`

This is a handoff/rail hardening change. It does not contact WING, start a
backend, write mixer state, grant live authority, or close the WING read-only
connection proof blocker.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  backend/wing_readonly_preflight_contract.py
  tools/build_wing_readonly_preflight_handoff_packet.py
  tools/no_wing_release_check.py
  tests/test_wing_readonly_preflight_handoff_packet.py
  tests/test_no_wing_release_check.py tests/test_project_director_agent.py
  tests/test_wing_mvp_rails.py`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_handoff_packet.py
  tests/test_no_wing_release_check.py --tb=short`: `265 passed`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_project_director_agent.py tests/test_codex_direction_guard.py
  tests/test_wing_mvp_rails.py --tb=short`: `23 passed`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py
  --quick --backend-port 65535 --artifact-root
  /tmp/automixer_goal_supervised_gate_expected_artifact_marker_20260614`:
  passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/codex_autoloop/project_director_agent.py --format json --print`:
  selected `WING read-only connection proof`, `guard_status: on_target`,
  `selected_evidence_file_count: 49`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/codex_autoloop/codex_direction_guard.py --format json`:
  `status: on_target`, `stop_the_line: false`, `score: 90`.
- `/opt/homebrew/bin/git diff --check`: passed.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: `3284 passed, 1 skipped, 1 warning`.
