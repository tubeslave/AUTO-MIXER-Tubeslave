# Plan M018: Integration policy contract

## Checkpoints

| Checkpoint | Scope | Files | Evidence requirement | Safety boundary | Validation | Exit condition | Rollback | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M018-CP01 | Add fail-closed integration policy contract | `backend/integration_policy_contract.py`, `tests/test_integration_policy_contract.py` | Candidate integration policies are complete and evidence-bound before owner review, and owner approval is required before policy use | Offline policy evidence only, no hardware, no OSC writes | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_integration_policy_contract.py -q` | contract result keeps `live_authority_granted: false` and `readiness_closure_allowed: false` | revert M018 commit | completed |
| M018-CP02 | Bind integration policy to roadmap and completion gates | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/PROJECT_COMPLETION_GATE.md`, `Docs/adr/integration-policy-contract-20260619.md`, `tests/test_wing_mvp_rails.py` | M018 remains pending until integration controls, evidence bindings and owner approval exist | Report-only gate binding | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m018_integration_policy_contract_stays_owner_approved_and_offline -q` | roadmap/capability/current-state/report rows | revert M018 commit | completed |

## Validation Contract

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_integration_policy_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m018_integration_policy_contract_stays_owner_approved_and_offline -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Acceptance Criteria

- Candidate integration policies cover analysis trace, offline replay binding,
  fault recovery binding, safety-kernel trace, operator report,
  reproducibility, cross-scenario consistency and no-live-transport controls.
- Integration evidence binds the policy to replay and fault manifests,
  threshold and recovery policy results, integration checks, operator report,
  reproducibility artifact, cross-scenario artifact and human supervision.
- Candidate evidence can become `ready_for_owner_review` without becoming
  usable for offline integration promotion.
- Owner approval requires explicit approval fields before integration policy
  use.
- Candidate integration policy evidence cannot grant live authority, close
  readiness, contact WING/backend, send live writes or become hardware proof.
