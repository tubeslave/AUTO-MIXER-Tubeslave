# Plan M106: Project Director Post-M105 Intake Loop Persistence

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M106-CP01 | Reproduce the M105 persistence gap | `tools/codex_autoloop/project_director_agent.py`, `tests/test_project_director_agent.py` | Project Director must still find the M095-M104 WING read-only guard streak when direction-guard commits sit above it | Read-only Project Director; no backend/WING/OSC contact | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_project_director_agent.py -q` | focused Project Director run | revert streak helper update | completed |
| M106-CP02 | Persist the post-M104 loop guard after M105 | `tools/codex_autoloop/project_director_agent.py`, `tests/test_project_director_agent.py` | The streak helper returns the longest WING read-only intake guard streak inside recent history instead of only the prefix streak | Control-plane only; no production mixing/runtime authority | focused Project Director run | focused run | revert helper update | completed |
| M106-CP03 | Record direction evidence without changing WING readiness | `Docs/ROADMAP.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md`, `ops/autopilot/STATE.json`, `ops/autopilot/RUN_LEDGER.jsonl`, `Docs/reports/AUTOMIXER_AUTOPILOT_M106_PROJECT_DIRECTOR_POST_M105_INTAKE_LOOP_PERSISTENCE_20260620.md` | M106 is recorded as a Project Director persistence fix while M001/M002 stay external-validation states | Report/control-plane only; no Full Autonomous readiness | WING rail, full suite, memory lint | M106 rail and full-suite evidence recorded after validation | revert state/report updates | completed |

## Required validation

- Project Director focused tests pass.
- M106 rail proves Project Director keeps `external_validation_wait_or_named_gap_only` after the M105 commit.
- `ops/autopilot/STATE.json` remains valid JSON.
- `ops/autopilot/RUN_LEDGER.jsonl` remains valid JSONL.
- Full suite passes in a clean staged-only worktree before commit.

## Boundary

M106 does not contact WING, backend or OSC transport. It does not grant live
authority, readiness closure, hardware evidence or Full Autonomous readiness.
It fixes the M105 direction guard so the guard remains active after its own
direction-guard commit sits above the WING read-only intake-hardening streak.
