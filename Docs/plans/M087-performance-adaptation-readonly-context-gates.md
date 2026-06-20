# Plan M087: Performance adaptation read-only context gates

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M087-CP01 | Gate performance-adaptation proposals on fresh no-authority context | `backend/performance_adaptation_proposal.py`, `tests/test_performance_adaptation_proposal.py` | Persistent musician-level drift can become a proposal only when current performance context is fresh, read-only and carries no live authority | Offline proposal-only; no backend/WING/OSC contact | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_performance_adaptation_proposal.py -q` | focused performance run: `14 passed` | revert performance-context gate changes | completed |
| M087-CP02 | Keep missing or live-authority performance context analysis-only | `backend/performance_adaptation_proposal.py`, `tests/test_performance_adaptation_proposal.py` | Missing, stale or live-authority context blocks proposal emission and keeps `can_apply_live` false | Proposal-only; no queue import or live apply | focused performance tests | `14 passed` | revert blocker/evidence changes | completed |
| M087-CP03 | Record performance context evidence in proposal-only contracts | `backend/processing_proposal_adapters.py`, `tests/test_performance_adaptation_proposal.py` | Performance-adaptation proposal contracts require `current_performance_context_cache_no_live_authority` evidence before promotion planning | Offline contract evidence only | focused performance tests | `14 passed` | revert contract-evidence addition | completed |
| M087-CP04 | Record goal-facing evidence without changing WING readiness | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md`, `ops/autopilot/STATE.json`, `ops/autopilot/RUN_LEDGER.jsonl`, `Docs/reports/AUTOMIXER_AUTOPILOT_M087_PERFORMANCE_ADAPTATION_READONLY_CONTEXT_GATES_20260620.md` | M087 is recorded as offline performance-adaptation hardening while M001/M002 stay external-validation states | Report-only; no Full Autonomous readiness | WING rail, full suite, memory lint | M087 rail `1 passed`; full WING rail `100 passed`; clean staged full suite `4000 passed, 1 skipped, 1 warning`; memory lint PASS | revert state/report updates | completed |

## Required validation at checkpoint close

- Performance-adaptation focused tests pass.
- M087 rail proves missing, stale or live-authority performance context blocks proposals and keeps live authority denied.
- Standard backend test command passes in a clean staged worktree.
- Memory lint passes.
- `ops/autopilot/STATE.json` and `ops/autopilot/RUN_LEDGER.jsonl` remain valid.
- No backend/WING/OSC transport, live write, proposal apply, readiness closure, hardware evidence or Full Autonomous readiness is created.
