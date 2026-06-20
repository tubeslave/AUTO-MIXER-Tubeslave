# Plan M094: Project Director External-Validation Scope Guard

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M094-CP01 | Detect selected WING actions that require external validation | `tools/codex_autoloop/project_director_agent.py`, `tests/test_project_director_agent.py` | Project Director marks the current selected action as `external_validation_wait` when the first open blocker requires real WING/operator evidence | Read-only Project Director; no backend/WING/OSC contact | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_project_director_agent.py -q` | focused Project Director run | revert Project Director scope helper | completed |
| M094-CP02 | Remove the broad proposal-only escape hatch while M001 waits | `tools/codex_autoloop/project_director_agent.py`, `tests/test_project_director_agent.py` | Allowed autonomous scope becomes `external_validation_handoff_or_control_plane_only`, and new proposal-only context-gate milestones are forbidden unless the selected WING blocker or operator asks for them | Control-plane only; no production mixing/runtime code | focused Project Director run | focused run | revert allowed/forbidden scope changes | completed |
| M094-CP03 | Record direction evidence without changing WING readiness | `Docs/ROADMAP.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md`, `ops/autopilot/STATE.json`, `ops/autopilot/RUN_LEDGER.jsonl`, `Docs/reports/AUTOMIXER_AUTOPILOT_M094_PROJECT_DIRECTOR_EXTERNAL_VALIDATION_SCOPE_GUARD_20260620.md` | M094 is recorded as an autopilot direction guard while M001/M002 stay external-validation states | Report/control-plane only; no Full Autonomous readiness | WING rail, full suite, memory lint | M094 rail and full-suite evidence recorded after validation | revert state/report updates | completed |

## Required validation at checkpoint close

- Project Director focused tests pass.
- M094 rail proves the current WING selected action narrows autonomous scope to
  external-validation handoff/intake or control-plane state.
- Standard backend test command passes in a clean staged worktree.
- Memory lint passes.
- `ops/autopilot/STATE.json` and `ops/autopilot/RUN_LEDGER.jsonl` remain valid.
- No backend/WING/OSC transport, live write, proposal apply, queue import,
  readiness closure, hardware evidence or Full Autonomous readiness is created.
