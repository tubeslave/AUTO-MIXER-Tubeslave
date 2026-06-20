# Plan M089: Mute management read-only context gates

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M089-CP01 | Gate mute proposals on fresh no-authority mute state | `backend/mute_management_proposal.py`, `tests/test_mute_management_proposal.py` | Mute/unmute candidates can become proposal-only recommendations only when current mute state context is fresh, read-only and carries no live authority | Offline proposal-only; no backend/WING/OSC contact | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_mute_management_proposal.py -q` | focused mute run: `15 passed` | revert mute-context gate changes | completed |
| M089-CP02 | Keep missing or live-authority mute context analysis-only | `backend/mute_management_proposal.py`, `tests/test_mute_management_proposal.py` | Missing, stale or live-authority mute context blocks proposal emission and keeps `can_apply_live` false | Proposal-only; no queue import or live apply | focused mute tests | `15 passed` | revert blocker/evidence changes | completed |
| M089-CP03 | Record mute context evidence in proposal-only contracts | `backend/processing_proposal_adapters.py`, `tests/test_mute_management_proposal.py` | Mute proposal contracts require `current_mute_cache_no_live_authority` evidence before promotion planning | Offline contract evidence only | focused mute tests | `15 passed` | revert contract-evidence addition | completed |
| M089-CP04 | Record goal-facing evidence without changing WING readiness | `Docs/ROADMAP.md`, `Docs/CONSOLE_CAPABILITY_MATRIX.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md`, `ops/autopilot/STATE.json`, `ops/autopilot/RUN_LEDGER.jsonl`, `Docs/reports/AUTOMIXER_AUTOPILOT_M089_MUTE_MANAGEMENT_READONLY_CONTEXT_GATES_20260620.md` | M089 is recorded as offline mute-control hardening while M001/M002 stay external-validation states | Report-only; no Full Autonomous readiness | WING rail, full suite, memory lint | M089 rail `1 passed`; full WING rail `102 passed`; clean staged full suite `4006 passed, 1 skipped, 1 warning`; memory lint PASS | revert state/report updates | completed |

## Required validation at checkpoint close

- Mute-management focused tests pass.
- M089 rail proves missing, stale or live-authority mute context blocks proposals and keeps live authority denied.
- Standard backend test command passes in a clean staged worktree.
- Memory lint passes.
- `ops/autopilot/STATE.json` and `ops/autopilot/RUN_LEDGER.jsonl` remain valid.
- No backend/WING/OSC transport, live write, proposal apply, readiness closure, hardware evidence or Full Autonomous readiness is created.
