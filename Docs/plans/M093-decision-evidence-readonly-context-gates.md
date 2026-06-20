# Plan M093: Decision evidence packet read-only context gates

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M093-CP01 | Gate decision-evidence packets on fresh no-authority decision context | `backend/decision_evidence_packet_proposal.py`, `tests/test_decision_evidence_packet_proposal.py` | Decision-evidence packet candidates can become proposal-only recommendations only when current decision context is fresh, read-only and carries no live authority | Offline proposal-only; no backend/WING/OSC contact | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_decision_evidence_packet_proposal.py -q` | focused decision-evidence run | revert decision-evidence context rail changes | completed |
| M093-CP02 | Require trusted source identity, readback state and rollback planning | `backend/decision_evidence_packet_proposal.py`, `backend/project_goal_capability_matrix.py`, `tests/test_project_goal_capability_matrix.py` | Console-affecting evidence packets require trusted source identity, current readback state and rollback planning before downstream proposals can trust them | Proposal-only; no queue import or live apply | focused decision-evidence and capability tests | focused runs | revert source/readback/rollback gate changes | completed |
| M093-CP03 | Keep missing, stale or live-authority decision context analysis-only | `backend/decision_evidence_packet_proposal.py`, `tests/test_decision_evidence_packet_proposal.py` | Missing, stale or live-authority decision context blocks proposal emission and keeps `can_apply_live` false | Proposal-only; no queue import or live apply | focused decision-evidence tests | focused run | revert blocker/evidence changes | completed |
| M093-CP04 | Record goal-facing evidence without changing WING readiness | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md`, `ops/autopilot/STATE.json`, `ops/autopilot/RUN_LEDGER.jsonl`, `Docs/reports/AUTOMIXER_AUTOPILOT_M093_DECISION_EVIDENCE_READONLY_CONTEXT_GATES_20260620.md` | M093 is recorded as offline decision-basis hardening while M001/M002 stay external-validation states | Report-only; no Full Autonomous readiness | WING rail, full suite, memory lint | M093 rail and full-suite evidence recorded after validation | revert state/report updates | completed |

## Required validation at checkpoint close

- Decision-evidence focused tests pass.
- Capability matrix focused validation passes.
- M093 rail proves missing, stale or live-authority decision context blocks
  proposals and keeps live authority denied.
- Standard backend test command passes in a clean staged worktree.
- Memory lint passes.
- `ops/autopilot/STATE.json` and `ops/autopilot/RUN_LEDGER.jsonl` remain valid.
- No backend/WING/OSC transport, live write, proposal apply, queue import,
  readiness closure, hardware evidence or Full Autonomous readiness is created.
