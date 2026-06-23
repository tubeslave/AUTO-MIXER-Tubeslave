# Automixer WING MVP Current Command Required-False Detail Match Flags

Date: 2026-06-14
Status: Validated

## Summary

Current First Command now requires the no-WING handoff summary to expose
packet-vs-plan match verdicts for the backend-port required-false detail fields.

## Why

The first WING blocker remains `WING read-only connection proof`. Offline
handoff packets must not imply permission to stop listeners, grant live
authority, execute the backend-port prestart gate, update readiness, or close
the blocker. The selected action baseline must therefore require explicit
`*_matches_plan: true` verdicts for those no-authority fields.

## Changed surfaces

- `tools/codex_autoloop/update_mvp_progress.py` adds the five verdicts to
  `CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS`.
- `.ai/project-layer/wing_mvp_blockers.md`, `Docs/MVP_WING_SUPERVISED.md`,
  `Docs/runbooks/WING_READONLY_PREFLIGHT.md`, `tools/codex_autoloop/README.md`,
  and `tools/codex_autoloop/codex_cycle_prompt.md` now require the same markers.
- `tests/test_project_director_agent.py` and `tests/test_wing_mvp_rails.py`
  assert selected-action, prompt/readme, report, and rail visibility.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from offline evidence.

## Validation

- Focused Project Director and WING rail pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
  --tb=short -x`: 11 passed in 0.37s.
- Project Director JSON selected `WING read-only connection proof`,
  `selected_blocker_status: tooling_ready_waits_real_wing`,
  `selected_action_source: Current First Command`,
  `current_first_command_baseline_coverage_marker_count: 104`,
  `current_first_command_baseline_coverage_passed: true`,
  `current_first_command_baseline_missing_markers: []`,
  `selected_evidence_file_count: 103`, `guard_status: on_target`,
  `guard_score: 90`, and `guard_stop_the_line: false`.
- no-WING quick without frontend:
  `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/no_wing_release_check.py --quick --skip-frontend --artifact-root
  /tmp/automixer_goal_current_command_required_false_matches_nowing`: passed.
  The handoff packet artifact reported `live_write: false`,
  `queries_console: false`, `safe_to_stop_listener_by_gate: false`,
  `live_authority_granted: false`, the three backend-port prestart gate fields
  as `false`, and all five corresponding `*_matches_plan` verdicts as `true`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  tools/codex_autoloop/update_mvp_progress.py`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`:
  PASS.
- `/opt/homebrew/bin/git diff --check`: passed.
- Standard full suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3307 passed, 9 skipped, 1 warning in 72.29s.
