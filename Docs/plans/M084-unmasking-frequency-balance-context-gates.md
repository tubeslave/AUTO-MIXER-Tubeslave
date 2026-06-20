# Plan M084: Unmasking, frequency and balance context gates

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M084-CP01 | Gate cross-channel unmasking on fresh no-authority masking and arrangement context | `backend/cross_adaptive_eq.py`, `tests/test_cross_adaptive_eq.py` | Linked cut/boost EQ proposals require fresh read-only/no-authority masking context, fresh read-only/no-authority arrangement context, and usable arrangement priorities | Offline proposal-only; no backend/WING/OSC contact | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_cross_adaptive_eq.py -q` | focused downstream proposal run: `36 passed` | revert cross-adaptive EQ gate changes | completed |
| M084-CP02 | Gate frequency separation on fresh no-authority spectrum context | `backend/frequency_separation_proposal.py`, `tests/test_frequency_separation_proposal.py` | Source/stem frequency-lane recommendations require fresh read-only/no-authority spectrum context before proposal emission | Offline proposal-only; no queue import or live apply | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_frequency_separation_proposal.py -q` | focused downstream proposal run: `36 passed` | revert frequency separation gate changes | completed |
| M084-CP03 | Gate bus/group balance on fresh no-authority bus/DCA/matrix context | `backend/bus_group_balance_proposal.py`, `tests/test_bus_group_balance_proposal.py` | Bus/group/DCA balance recommendations require fresh read-only/no-authority bus state before proposal emission | Offline proposal-only; bus/DCA writes remain unapproved | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_bus_group_balance_proposal.py -q` | focused downstream proposal run: `36 passed` | revert bus/group balance gate changes | completed |
| M084-CP04 | Record goal-facing evidence without changing WING readiness | `Docs/ROADMAP.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md`, `ops/autopilot/STATE.json`, `ops/autopilot/RUN_LEDGER.jsonl`, `Docs/reports/AUTOMIXER_AUTOPILOT_M084_UNMASKING_FREQUENCY_BALANCE_CONTEXT_GATES_20260620.md` | M084 is recorded as offline downstream mix-decision hardening while M001/M002 stay external-validation states | Report-only; no Full Autonomous readiness | WING rail, full suite, memory lint | M084 rail, full suite, memory lint and JSON/JSONL validation | revert state/report updates | completed |

## Required validation at checkpoint close

- Cross-adaptive EQ, frequency separation and bus/group balance focused tests pass.
- M084 rail proves missing, stale or live-authority context blocks downstream proposals and keeps live authority denied.
- Standard backend test command passes.
- Memory lint passes.
- `ops/autopilot/STATE.json` and `ops/autopilot/RUN_LEDGER.jsonl` remain valid.
- No backend/WING/OSC transport, live write, proposal apply, readiness closure, hardware evidence or Full Autonomous readiness is created.
