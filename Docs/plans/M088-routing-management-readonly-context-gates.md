# Plan M088: Routing management read-only context gates

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M088-CP01 | Gate routing proposals on fresh no-authority route state | `backend/routing_management_proposal.py`, `tests/test_routing_management_proposal.py` | Routing diffs can become proposal-only recommendations only when current route readback context is fresh, read-only and carries no live authority | Offline proposal-only; no backend/WING/OSC contact | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_routing_management_proposal.py -q` | focused routing run: `14 passed` | revert routing-context gate changes | completed |
| M088-CP02 | Keep missing or live-authority route context analysis-only | `backend/routing_management_proposal.py`, `tests/test_routing_management_proposal.py` | Missing, stale or live-authority route context blocks routing proposal emission and keeps `can_apply_live` false | Proposal-only; no queue import or live apply | focused routing tests | `14 passed` | revert blocker/evidence changes | completed |
| M088-CP03 | Record route context evidence in proposal-only contracts | `backend/processing_proposal_adapters.py`, `tests/test_routing_management_proposal.py` | Routing proposal contracts require `current_route_cache_no_live_authority` evidence before promotion planning | Offline contract evidence only | focused routing tests | `14 passed` | revert contract-evidence addition | completed |
| M088-CP04 | Record goal-facing evidence without changing WING readiness | `Docs/ROADMAP.md`, `Docs/CONSOLE_CAPABILITY_MATRIX.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md`, `ops/autopilot/STATE.json`, `ops/autopilot/RUN_LEDGER.jsonl`, `Docs/reports/AUTOMIXER_AUTOPILOT_M088_ROUTING_MANAGEMENT_READONLY_CONTEXT_GATES_20260620.md` | M088 is recorded as offline routing-control hardening while M001/M002 stay external-validation states | Report-only; no Full Autonomous readiness | WING rail, full suite, memory lint | M088 rail `1 passed`; full WING rail `101 passed`; clean staged full suite `4003 passed, 1 skipped, 1 warning`; memory lint PASS | revert state/report updates | completed |

## Required validation at checkpoint close

- Routing-management focused tests pass.
- M088 rail proves missing, stale or live-authority route context blocks proposals and keeps live authority denied.
- Standard backend test command passes in a clean staged worktree.
- Memory lint passes.
- `ops/autopilot/STATE.json` and `ops/autopilot/RUN_LEDGER.jsonl` remain valid.
- No backend/WING/OSC transport, live write, proposal apply, readiness closure, hardware evidence or Full Autonomous readiness is created.
