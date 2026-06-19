# Plan M016: Recovery policy contract

## Checkpoints

| Checkpoint | Scope | Files | Evidence requirement | Safety boundary | Validation | Exit condition | Rollback | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M016-CP01 | Add fail-closed recovery policy contract | `backend/recovery_policy_contract.py`, `tests/test_recovery_policy_contract.py` | Candidate recovery policies are complete and evidence-bound before owner review, and owner approval is required before policy use | Offline policy evidence only, no hardware, no OSC writes | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_recovery_policy_contract.py -q` | contract result keeps `live_authority_granted: false` and `readiness_closure_allowed: false` | revert M016 commit | completed |
| M016-CP02 | Bind recovery policy to roadmap and completion gates | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/PROJECT_COMPLETION_GATE.md`, `Docs/adr/recovery-policy-contract-20260619.md`, `tests/test_wing_mvp_rails.py` | M016 remains pending until recovery controls, evidence bindings and owner approval exist | Report-only gate binding | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m016_recovery_policy_contract_stays_owner_approved_and_offline -q` | roadmap/capability/current-state/report rows | revert M016 commit | completed |

## Validation Contract

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_recovery_policy_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m016_recovery_policy_contract_stays_owner_approved_and_offline -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Acceptance Criteria

- Candidate recovery policies cover fault detection, fail-closed behavior,
  unsafe-action blocking, safety-kernel enforcement, rollback, emergency stop,
  operator visibility, audit logging and no-live-transport controls.
- Recovery evidence binds the policy to a fault scenario matrix, rollback
  reference, operator message template, audit schema, stale-state timeout,
  connection-loss timeout, emergency-stop procedure and human supervision.
- Candidate evidence can become `ready_for_owner_review` without becoming
  usable for simulator fault-recovery promotion.
- Owner approval requires explicit approval fields before recovery policy use.
- Candidate recovery policy evidence cannot grant live authority, close
  readiness, contact WING/backend, send live writes or become hardware proof.
