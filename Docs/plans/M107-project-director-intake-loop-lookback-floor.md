# Plan M107: Project Director Intake Loop Lookback Floor

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M107-CP01 | Reproduce the post-M106 lookback horizon gap | `tools/codex_autoloop/project_director_agent.py`, `tests/test_project_director_agent.py` | Project Director must still find the M095-M104 WING read-only guard streak after additional direction-guard commits sit above it | Read-only Project Director; no backend/WING/OSC contact | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_project_director_agent.py -q` | focused Project Director run | revert lookback-floor helper | completed |
| M107-CP02 | Add a minimum effective lookback for normal Project Director runs | `tools/codex_autoloop/project_director_agent.py`, `tests/test_project_director_agent.py` | A normal/default `lookback_commits=12` request is raised to the intake-loop guard floor before repo evidence is collected | Control-plane only; no production mixing/runtime authority | focused Project Director run | focused run | revert effective lookback helper | completed |
| M107-CP03 | Record direction evidence without changing WING readiness | `Docs/ROADMAP.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md`, `ops/autopilot/STATE.json`, `ops/autopilot/RUN_LEDGER.jsonl`, `Docs/reports/AUTOMIXER_AUTOPILOT_M107_PROJECT_DIRECTOR_INTAKE_LOOP_LOOKBACK_FLOOR_20260620.md` | M107 is recorded as a Project Director lookback-floor fix while M001/M002 stay external-validation states | Report/control-plane only; no Full Autonomous readiness | WING rail, full suite, memory lint | M107 rail and full-suite evidence recorded after validation | revert state/report updates | completed |

## Required Validation

- Project Director focused tests pass.
- M107 rail proves Project Director keeps `external_validation_wait_or_named_gap_only` with the requested `lookback_commits=12`.
- `ops/autopilot/STATE.json` remains valid JSON.
- `ops/autopilot/RUN_LEDGER.jsonl` remains valid JSONL.
- Full suite passes in a clean staged-only worktree before commit.

## Boundary

M107 does not contact WING, backend or OSC transport. It does not grant live
authority, readiness closure, hardware evidence or Full Autonomous readiness.
It fixes the Project Director evidence window so the M105/M106 loop guard does
not turn off after a small number of direction-guard commits push the WING
read-only streak outside the requested lookback window.
