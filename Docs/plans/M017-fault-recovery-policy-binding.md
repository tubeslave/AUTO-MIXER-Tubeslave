# Plan M017: Fault recovery policy binding

## Checkpoints

| Checkpoint | Scope | Files | Evidence requirement | Safety boundary | Validation | Exit condition | Rollback | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M017-CP01 | Bind fault-recovery promotion to the recovery-policy contract | `backend/fault_injection_recovery_contract.py`, `tests/test_fault_injection_recovery_contract.py`, `tests/test_evaluation_integration_bundle_contract.py` | Simulator fault-recovery manifests must include a ready M016 `recovery_policy_manifest`; loose legacy `recovery_policy` fields are summary context only | Offline simulator evidence only, no hardware, no OSC writes | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_fault_injection_recovery_contract.py -q` | legacy policy-only manifests remain blocked; bound manifest passes only offline | revert M017 commit | completed |
| M017-CP02 | Bind M017 to roadmap, completion gates and autopilot state | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/PROJECT_COMPLETION_GATE.md`, `Docs/adr/fault-recovery-policy-binding-20260619.md`, `tests/test_wing_mvp_rails.py` | M017 remains pending until simulator evidence uses the recovery policy contract result | Report-only gate binding | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m017_fault_recovery_policy_binding_requires_m016_manifest -q` | roadmap/capability/current-state/report rows | revert M017 commit | completed |

## Validation Contract

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_fault_injection_recovery_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_evaluation_integration_bundle_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m017_fault_recovery_policy_binding_requires_m016_manifest -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Acceptance Criteria

- Fault-recovery manifests consume `recovery_policy_manifest` from M016 before
  reporting simulator fault-recovery promotion readiness.
- Legacy `recovery_policy` fields without the M016 contract manifest remain
  visible as summary context but cannot satisfy approved-recovery-policy
  requirements.
- Integration bundles surface the same policy-binding failure through the
  `fault_recovery:` child prefix.
- Bound recovery policy manifests still cannot grant live authority, close
  readiness, contact WING/backend, send live writes or become hardware proof.
