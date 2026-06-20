# Plan M105: Project Director Post-M104 Intake Loop Guard

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M105-CP01 | Detect repeated WING read-only intake hardening after M104 | `tools/codex_autoloop/project_director_agent.py`, `tests/test_project_director_agent.py` | Project Director derives the consecutive WING read-only guard streak from recent commit subjects and flags the post-M104 intake-hardening loop | Read-only Project Director; no backend/WING/OSC contact | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_project_director_agent.py -q` | focused Project Director run | revert helper and directive fields | completed |
| M105-CP02 | Narrow post-M104 autonomous scope | `tools/codex_autoloop/project_director_agent.py`, `tests/test_project_director_agent.py` | While M001 waits for external validation, allowed autonomous scope becomes `external_validation_wait_or_named_gap_only` after the M095-M104 intake-hardening streak | Control-plane only; no production mixing/runtime authority | focused Project Director run | focused run | revert allowed/forbidden scope changes | completed |
| M105-CP03 | Record direction evidence without changing WING readiness | `Docs/ROADMAP.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md`, `ops/autopilot/STATE.json`, `ops/autopilot/RUN_LEDGER.jsonl`, `Docs/reports/AUTOMIXER_AUTOPILOT_M105_PROJECT_DIRECTOR_POST_M104_INTAKE_LOOP_GUARD_20260620.md` | M105 is recorded as an autopilot direction guard while M001/M002 stay external-validation states | Report/control-plane only; no Full Autonomous readiness | WING rail, full suite, memory lint | M105 rail and full-suite evidence recorded after validation | revert state/report updates | completed |

## Required validation

- Project Director focused tests pass.
- M105 rail proves Project Director detects the post-M104 WING read-only intake-hardening streak.
- M105 rail proves further autonomous work is narrowed to real external validation, explicit operator request or a named unverified contract gap.
- `ops/autopilot/STATE.json` remains valid JSON.
- `ops/autopilot/RUN_LEDGER.jsonl` remains valid JSONL.
- Full test suite passes in the staged-only worktree before commit.

## Boundary

M105 does not contact WING, backend or OSC transport. It does not grant live
authority, readiness closure, hardware evidence or Full Autonomous readiness.
It prevents the autonomous loop from treating another generic WING read-only
intake hardening pass as meaningful progress after M104 unless new evidence or
a named contract gap exists.
