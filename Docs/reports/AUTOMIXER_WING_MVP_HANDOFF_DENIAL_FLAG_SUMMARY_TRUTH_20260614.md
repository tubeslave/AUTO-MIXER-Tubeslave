# WING MVP Handoff Denial Flag Summary Truth

## Summary

The WING read-only handoff summary now preserves actual unsafe `true` values for
packet execution/readiness/live-authority flags when a handoff packet fails
validation. A failed packet can no longer be rendered with hardcoded safe
`false` values for these operator-facing fields.

The summary artifact and Markdown must reflect actual unsafe values for:

- `execution_allowed_by_packet`
- `readiness_matrix_update_allowed_by_packet`
- `readiness_closure_allowed_by_packet`
- `live_authority_granted`
- `full_autonomous_ready`

This is no-WING audit hardening. It does not contact WING, start a backend,
write mixer state, grant live authority, or close the WING read-only connection
proof blocker.

## Validation

Passed before commit:

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_unsafe_wing_preflight_handoff_packet_artifact tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short`
  - Result: `264 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --backend-port 65535 --artifact-root /tmp/automixer_goal_handoff_denial_flag_summary_truth_20260614`
  - Result: passed.
  - Safe path summary preserved all five denial/live flags as `false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json --output .paperclip/reports/codex_direction_guard_final.md`
  - Result: `status=on_target`, `score=90`, `stop_the_line=false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --output .paperclip/reports/project_director_final.md --print`
  - Result: selected blocker `WING read-only connection proof`, `guard_status=on_target`, `guard_score=90`, `guard_stop_the_line=false`, evidence file count `53`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  - Result: `3289 passed, 1 skipped, 1 warning`.
