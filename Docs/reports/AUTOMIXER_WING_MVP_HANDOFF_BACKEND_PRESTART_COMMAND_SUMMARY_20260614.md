# WING MVP Handoff Backend Prestart Command Summary

## Summary

The WING read-only handoff summary now exposes the packet-declared
`backend_port_prestart_check_command` in both JSON and Markdown output. This is
shown beside `prestart_backend_port_check_command`, which is extracted from the
handoff command map.

This keeps the operator-facing no-WING handoff summary aligned with the Project
Director rail that requires the backend prestart command to exactly match:

`PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`

Failed handoff packets preserve mismatched command values in the summary so the
operator can see the exact bad command before any backend/WING session starts.

This is no-WING audit hardening. It does not contact WING, start a backend,
write mixer state, grant live authority, or close the WING read-only connection
proof blocker.

## Validation

Passed before commit:

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_wing_preflight_handoff_packet_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_unsafe_wing_preflight_handoff_packet_artifact tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short`
  - Result: `264 passed`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --backend-port 65535 --artifact-root /tmp/automixer_goal_handoff_backend_prestart_command_summary_20260614`
  - Result: passed.
  - Safe path summary showed `backend_port_prestart_check_command` and `prestart_backend_port_check_command` both matching `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json --output .paperclip/reports/codex_direction_guard_backend_prestart_command_summary.md`
  - Result: `status=on_target`, `score=90`, `stop_the_line=false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --output .paperclip/reports/project_director_backend_prestart_command_summary.md --print`
  - Result: selected blocker `WING read-only connection proof`, `guard_status=on_target`, `guard_score=90`, `guard_stop_the_line=false`, evidence file count `55`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  - Result: `3289 passed, 1 skipped, 1 warning`.
