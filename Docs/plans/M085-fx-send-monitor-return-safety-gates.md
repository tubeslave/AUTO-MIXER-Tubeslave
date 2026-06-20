# Plan M085: FX send monitor and return safety gates

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M085-CP01 | Gate FX send proposals on fresh no-authority send/return context | `backend/fx_send_proposal.py`, `tests/test_fx_send_proposal.py` | FX send recommendations require fresh read-only/no-authority send/return state before proposal emission | Offline proposal-only; no backend/WING/OSC contact | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_fx_send_proposal.py -q` | focused FX run: `24 passed` | revert FX send context gate changes | completed |
| M085-CP02 | Require mapped no-authority FX parameter schema and rollback/readback evidence | `backend/fx_send_proposal.py`, `backend/processing_proposal_adapters.py`, `tests/test_fx_send_proposal.py` | FX proposals carry old-value and rollback-value sources and block missing, generic, mismatched or live-authority parameter schemas | Proposal-only; no queue import or live apply | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_fx_send_proposal.py -q` | focused FX run: `24 passed` | revert FX schema/readback contract changes | completed |
| M085-CP03 | Gate monitor exposure and FX return level safety | `backend/fx_send_proposal.py`, `tests/test_fx_send_proposal.py` | Monitor-exposed sends require fresh monitor-safety context; high monitor risk, feedback risk, intelligibility risk or return headroom failure blocks proposals | Offline analysis only for unsafe cases | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_fx_send_proposal.py -q` | focused FX run: `24 passed` | revert monitor/return safety checks | completed |
| M085-CP04 | Record goal-facing evidence without changing WING readiness | `Docs/ROADMAP.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md`, `ops/autopilot/STATE.json`, `ops/autopilot/RUN_LEDGER.jsonl`, `Docs/reports/AUTOMIXER_AUTOPILOT_M085_FX_SEND_MONITOR_RETURN_SAFETY_GATES_20260620.md` | M085 is recorded as offline effects/send hardening while M001/M002 stay external-validation states | Report-only; no Full Autonomous readiness | WING rail, full suite, memory lint | `98 passed`; clean staged full suite `3991 passed, 1 skipped, 1 warning`; memory lint PASS | revert state/report updates | completed |

## Required validation at checkpoint close

- FX send focused tests pass.
- M085 rail proves missing, stale or live-authority FX context blocks proposals and keeps live authority denied.
- Standard backend test command passes.
- Memory lint passes.
- `ops/autopilot/STATE.json` and `ops/autopilot/RUN_LEDGER.jsonl` remain valid.
- No backend/WING/OSC transport, live write, proposal apply, readiness closure, hardware evidence or Full Autonomous readiness is created.
