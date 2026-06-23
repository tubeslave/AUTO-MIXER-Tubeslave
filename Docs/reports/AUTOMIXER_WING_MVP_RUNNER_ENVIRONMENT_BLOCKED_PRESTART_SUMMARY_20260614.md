# Automixer WING MVP Runner Environment-Blocked Prestart Summary

Date: 2026-06-14
Status: Validated

## Summary

The WING read-only preflight runner now preserves no-WING
`environment_blocked_checks` and `environment_blocked_check_details` when a
prestart summary is failed because the frontend dev server could not bind
`127.0.0.1:3000`.

## Why

The runner is the boundary before opening the backend/WING WebSocket. An
environment-blocked no-WING summary must not become usable WING proof, but the
blocked runner summary should explain the exact prestart failure so an operator
can rerun the no-WING gate in the right environment.

## Changed surfaces

- `tools/wing_readonly_preflight_runner.py` copies environment-blocked fields
  from the prestart summary into the blocked runner summary.
- The runner records `prestart_summary_environment_blocked` before the generic
  `prestart_summary_status_not_passed` violation.
- The runner's `operator_next_step` now names `127.0.0.1:3000`, requires a
  rerun of `tools/no_wing_release_check.py --quick`, and states that the
  environment-blocked summary grants no readiness or live authority.
- `tests/test_wing_readonly_preflight_runner.py` covers the fail-closed path
  and asserts the WebSocket is not opened.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from an environment-blocked no-WING prestart summary.

## Validation

- Focused runner/rails/Project Director/no-WING pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_runner.py
  tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
  tests/test_no_wing_release_check.py --tb=short -x`: 317 passed, 8 skipped
  in 3.34s.
- Project Director JSON selected `WING read-only connection proof`,
  `selected_blocker_status: tooling_ready_waits_real_wing`,
  `selected_action_source: Current First Command`,
  `current_first_command_baseline_coverage_passed: true`,
  `current_first_command_baseline_missing_markers: []`,
  `current_first_command_baseline_coverage_marker_count: 108`,
  `selected_evidence_file_count: 108`, included this report in
  `selected_evidence_files`, `guard_status: on_target`, `guard_score: 90`, and
  `guard_stop_the_line: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  tools/wing_readonly_preflight_runner.py
  tools/codex_autoloop/update_mvp_progress.py tools/no_wing_release_check.py`:
  passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`:
  PASS.
- `/opt/homebrew/bin/git diff --check`: passed.
- Full standard suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3316 passed, 9 skipped, 1 warning in 71.09s.
