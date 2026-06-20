# Plan M090: Pan spatial read-only context gates

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M090-CP01 | Gate pan proposals on fresh no-authority pan state | `backend/pan_spatial_proposal.py`, `tests/test_pan_spatial_proposal.py` | Pan candidates can become proposal-only recommendations only when current pan context is fresh, read-only and carries no live authority | Offline proposal-only; no backend/WING/OSC contact | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_pan_spatial_proposal.py -q` | focused pan run: `16 passed` | revert pan-context rail changes | completed |
| M090-CP02 | Gate non-center pan on fresh mono/phase safety context | `backend/pan_spatial_proposal.py`, `tests/test_pan_spatial_proposal.py` | Non-center pan candidates require fresh read-only/no-authority mono/phase evidence and safe mono sum before proposal emission | Proposal-only; no queue import or live apply | focused pan tests | `16 passed` | revert pan mono/phase rail changes | completed |
| M090-CP03 | Keep missing or live-authority pan context analysis-only | `backend/pan_spatial_proposal.py`, `tests/test_pan_spatial_proposal.py` | Missing, stale or live-authority pan or mono/phase context blocks proposal emission and keeps `can_apply_live` false | Proposal-only; no queue import or live apply | focused pan tests | `16 passed` | revert blocker/evidence changes | completed |
| M090-CP04 | Record goal-facing evidence without changing WING readiness | `Docs/ROADMAP.md`, `Docs/CONSOLE_CAPABILITY_MATRIX.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md`, `ops/autopilot/STATE.json`, `ops/autopilot/RUN_LEDGER.jsonl`, `Docs/reports/AUTOMIXER_AUTOPILOT_M090_PAN_SPATIAL_READONLY_CONTEXT_GATES_20260620.md` | M090 is recorded as offline pan-control hardening while M001/M002 stay external-validation states | Report-only; no Full Autonomous readiness | WING rail, full suite, memory lint | M090 rail `1 passed`; full WING rail `103 passed`; clean staged full suite `4007 passed, 1 skipped, 1 warning`; memory lint PASS | revert state/report updates | completed |

## Required validation at checkpoint close

- Pan-spatial focused tests pass.
- M090 rail proves missing, stale or live-authority pan/mono-phase context blocks proposals and keeps live authority denied.
- Standard backend test command passes in a clean staged worktree.
- Memory lint passes.
- `ops/autopilot/STATE.json` and `ops/autopilot/RUN_LEDGER.jsonl` remain valid.
- No backend/WING/OSC transport, live write, proposal apply, readiness closure, hardware evidence or Full Autonomous readiness is created.
