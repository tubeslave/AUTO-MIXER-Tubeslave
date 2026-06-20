# Plan M086: Master bus PA feed and reference safety gates

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M086-CP01 | Gate master-bus proposals on fresh no-authority main-bus state | `backend/master_bus_optimization_proposal.py`, `tests/test_master_bus_optimization_proposal.py` | Master/output recommendations require fresh read-only/no-authority main-bus processing context before proposal emission | Offline proposal-only; no backend/WING/OSC contact | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_master_bus_optimization_proposal.py -q` | focused master run: `14 passed` | revert master-bus context gate changes | completed |
| M086-CP02 | Require PA-feed readback and PA-feed safety evidence | `backend/master_bus_optimization_proposal.py`, `backend/processing_proposal_adapters.py`, `tests/test_master_bus_optimization_proposal.py` | Missing, stale or live-authority PA-feed context keeps master/output proposals analysis-only | Proposal-only; no queue import or live apply | focused master tests | `14 passed` | revert PA-feed safety contract changes | completed |
| M086-CP03 | Exclude reference/capture feeds from master-output proposals | `backend/master_bus_optimization_proposal.py`, `tests/test_master_bus_optimization_proposal.py` | Reference, measurement, capture and analysis feeds cannot become master-output proposal candidates | Offline analysis only for reference feeds | focused master tests | `14 passed` | revert reference-feed exclusion changes | completed |
| M086-CP04 | Record goal-facing evidence without changing WING readiness | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md`, `ops/autopilot/STATE.json`, `ops/autopilot/RUN_LEDGER.jsonl`, `Docs/reports/AUTOMIXER_AUTOPILOT_M086_MASTER_BUS_PA_FEED_REFERENCE_SAFETY_GATES_20260620.md` | M086 is recorded as offline master-bus/output hardening while M001/M002 stay external-validation states | Report-only; no Full Autonomous readiness | WING rail, full suite, memory lint | `99 passed`; clean staged full suite `3997 passed, 1 skipped, 1 warning`; memory lint PASS | revert state/report updates | completed |

## Required validation at checkpoint close

- Master-bus focused tests pass.
- M086 rail proves missing, stale or live-authority main/PA-feed context blocks proposals and keeps live authority denied.
- Standard backend test command passes in a clean staged worktree.
- Memory lint passes.
- `ops/autopilot/STATE.json` and `ops/autopilot/RUN_LEDGER.jsonl` remain valid.
- No backend/WING/OSC transport, live write, proposal apply, readiness closure, hardware evidence or Full Autonomous readiness is created.
