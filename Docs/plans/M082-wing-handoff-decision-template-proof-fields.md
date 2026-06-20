# Plan M082: WING handoff decision template proof fields

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M082-CP01 | Surface decision-template proof fields in handoff | `tools/build_wing_readonly_preflight_handoff_packet.py`, `tests/test_wing_readonly_preflight_handoff_packet.py` | Offline handoff packets publish the full review-decision template required field list and the prestart/verifier proof-field subset that operator edits must preserve | No backend, no WING, no OSC, no readiness closure | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_handoff_packet.py -q` | `21 passed` | revert M082 code/test changes | completed |
| M082-CP02 | Bind operator runbook to the new handoff surface | `Docs/runbooks/WING_READONLY_PREFLIGHT.md` | The real-run review instructions tell the operator to preserve `decision_template_prestart_proof_fields` from the generated template | Documentation only | rail test | `tests/test_wing_mvp_rails.py::test_m082_wing_handoff_surfaces_decision_template_proof_fields`: `1 passed` | revert runbook note | completed |
| M082-CP03 | Record evidence without changing readiness | `Docs/ROADMAP.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md`, `ops/autopilot/STATE.json`, `ops/autopilot/RUN_LEDGER.jsonl`, `Docs/reports/AUTOMIXER_AUTOPILOT_M082_WING_HANDOFF_DECISION_TEMPLATE_PROOF_FIELDS_20260620.md` | M082 is recorded as offline handoff hardening while M001/M002 stay `WAITING_FOR_EXTERNAL_VALIDATION` | Report-only | WING rail, full suite, memory lint | WING rails `95 passed`; full suite `4210 passed, 1 skipped, 1 warning`; memory lint PASS | revert state/report updates | completed |

## Required validation at checkpoint close

- Handoff packet unit tests pass.
- M082 rail proves the handoff surface, runbook note and state reports stay offline.
- Standard backend test command passes.
- Memory lint passes.
- `ops/autopilot/STATE.json` and `ops/autopilot/RUN_LEDGER.jsonl` remain valid.
- No WING/backend transport, OSC write, proposal apply, live authority, readiness closure or hardware evidence is created.
