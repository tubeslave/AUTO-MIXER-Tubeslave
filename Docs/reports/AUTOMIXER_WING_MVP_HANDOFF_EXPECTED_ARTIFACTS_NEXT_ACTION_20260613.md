# Automixer WING MVP Handoff Expected Artifacts Next Action

Status: Validated

## Summary

The WING MVP rails now require the no-WING handoff summary to show:

- `expected_artifacts` exactly matching `expected_artifact_globs`
- `expected_artifacts` exactly matching `plan_expected_artifacts`
- `expected_artifacts_match_plan: true`
- `summary_glob_matches_plan: true`
- `event_log_glob_matches_plan: true`

before any supervised fader/gain proof planning.

## Why

The handoff packet can only guide a later real WING read-only preflight when
its expected artifact list, named summary/events globs, and expected plan
artifact list all point at the same real-run outputs.

## Changed surfaces

- `.ai/project-layer/wing_mvp_blockers.md` requires the expected-artifacts
  fields in the first blocker next action and current command.
- `Docs/MVP_WING_SUPERVISED.md` lists missing expected-artifacts alignment as a
  stop-the-line condition.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` requires the fields in handoff
  validation.
- `tools/codex_autoloop/README.md` and
  `tools/codex_autoloop/codex_cycle_prompt.md` require future cycles to verify
  the fields.
- `Docs/INDEX.md` points to this report and the underlying contract report.
- `tests/test_project_director_agent.py` and `tests/test_wing_mvp_rails.py`
  cover the new rail.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/codex_autoloop/update_mvp_progress.py --blockers
  .ai/project-layer/wing_mvp_blockers.md --output
  Docs/reports/MVP_PROGRESS_LATEST.md`: passed.
- Focused Project Director/WING rail tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
  tests/test_codex_direction_guard.py --tb=short`: `23 passed`.
- Project Director JSON:
  `selected_blocker: WING read-only connection proof`,
  `selected_action` includes `expected_artifacts`,
  `expected_artifacts_match_plan: true`, `summary_glob_matches_plan: true`,
  and `event_log_glob_matches_plan: true`; `guard_status: on_target`;
  `guard_stop_the_line: false`.
- Direction guard:
  `status: on_target`, `current_focus: supervised_wing_rack_mvp`,
  `stop_the_line: false`, `score: 90`.
- `/opt/homebrew/bin/git diff --check`: passed.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- Compile check:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  tools/codex_autoloop/update_mvp_progress.py
  tools/codex_autoloop/project_director_agent.py
  tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`: passed.
- Full suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: `3280 passed, 1 skipped, 1 warning`.

## Boundary

This is an offline rail-hardening change. It does not start the backend, contact
WING, query the console, mutate mixer state, grant live authority, or close the
read-only WING proof blocker.
