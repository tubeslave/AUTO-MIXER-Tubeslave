# Plan M014: Quality threshold policy contract

## Checkpoints

| Checkpoint | Scope | Files | Evidence requirement | Safety boundary | Validation | Exit condition | Rollback | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M014-CP01 | Add fail-closed quality threshold policy contract | `backend/quality_threshold_policy_contract.py`, `tests/test_quality_threshold_policy_contract.py` | Candidate threshold policies are complete and evidence-bound before owner review, and owner approval is required before threshold use | Offline policy evidence only, no hardware, no OSC writes | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_quality_threshold_policy_contract.py -q` | contract result keeps `live_authority_granted: false` and `readiness_closure_allowed: false` | revert M014 commit | completed |
| M014-CP02 | Bind threshold policy to roadmap and completion gates | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/PROJECT_COMPLETION_GATE.md`, `Docs/adr/quality-threshold-policy-contract-20260619.md`, `tests/test_wing_mvp_rails.py` | M014 remains pending until required threshold axes, policy checks and owner approval exist | Report-only gate binding | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m014_quality_threshold_policy_contract_stays_owner_approved_and_offline -q` | roadmap/capability/current-state/report rows | revert M014 commit | completed |

## Validation Contract

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_quality_threshold_policy_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m014_quality_threshold_policy_contract_stays_owner_approved_and_offline -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Acceptance Criteria

- Candidate threshold policies cover every required quality axis from the replay
  contract.
- Each threshold axis has a metric, direction, candidate value, evidence
  artifact, rationale and rollback policy.
- Policy checks bind the candidate to baseline, reference, replay,
  fault-recovery and integration evidence.
- Candidate evidence can become `ready_for_owner_review` without becoming
  usable for replay or integration promotion.
- Owner approval requires explicit approval fields before threshold use.
- Candidate threshold policy evidence cannot grant live authority, close
  readiness, contact WING/backend, send live writes or become hardware proof.
