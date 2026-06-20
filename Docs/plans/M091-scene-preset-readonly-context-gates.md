# Plan M091: Scene/preset management read-only context gates

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M091-CP01 | Gate scene/preset proposals on fresh no-authority scene/preset state | `backend/scene_preset_management_proposal.py`, `tests/test_scene_preset_management_proposal.py` | Scene/preset candidates can become proposal-only recommendations only when current scene/preset context is fresh, read-only and carries no live authority | Offline proposal-only; no backend/WING/OSC contact | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_scene_preset_management_proposal.py -q` | focused scene/preset run: `22 passed` | revert scene/preset context rail changes | completed |
| M091-CP02 | Require scoped diffs and rollback evidence for destructive scene decisions | `backend/scene_preset_management_proposal.py`, `tests/test_scene_preset_management_proposal.py` | Scene recall/save and preset recall/save candidates require preview, scoped diff, rollback snapshot and destructive-scope limits before proposal emission | Proposal-only; no queue import or live apply | focused scene/preset tests | `22 passed` | revert scoped-diff and rollback gate changes | completed |
| M091-CP03 | Keep missing, stale or live-authority scene/preset context analysis-only | `backend/scene_preset_management_proposal.py`, `tests/test_scene_preset_management_proposal.py` | Missing, stale or live-authority current scene/preset context blocks proposal emission and keeps `can_apply_live` false | Proposal-only; no queue import or live apply | focused scene/preset tests | `22 passed` | revert blocker/evidence changes | completed |
| M091-CP04 | Record goal-facing evidence without changing WING readiness | `Docs/ROADMAP.md`, `Docs/CONSOLE_CAPABILITY_MATRIX.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md`, `ops/autopilot/STATE.json`, `ops/autopilot/RUN_LEDGER.jsonl`, `Docs/reports/AUTOMIXER_AUTOPILOT_M091_SCENE_PRESET_READONLY_CONTEXT_GATES_20260620.md` | M091 is recorded as offline scene/preset-control hardening while M001/M002 stay external-validation states | Report-only; no Full Autonomous readiness | WING rail, full suite, memory lint | M091 rail `1 passed`; full WING rail and full suite recorded in state after validation | revert state/report updates | completed |

## Required validation at checkpoint close

- Scene/preset focused tests pass.
- M091 rail proves missing, stale or live-authority scene/preset context blocks proposals and keeps live authority denied.
- Standard backend test command passes in a clean staged worktree.
- Memory lint passes.
- `ops/autopilot/STATE.json` and `ops/autopilot/RUN_LEDGER.jsonl` remain valid.
- No backend/WING/OSC transport, live write, proposal apply, readiness closure, hardware evidence or Full Autonomous readiness is created.
