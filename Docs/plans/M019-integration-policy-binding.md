# Plan M019: Integration policy binding

## Checkpoints

| Checkpoint | Scope | Files | Evidence requirement | Safety boundary | Validation | Exit condition | Rollback | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M019-CP01 | Bind offline integration promotion to the integration-policy contract | `backend/evaluation_integration_bundle_contract.py`, `tests/test_evaluation_integration_bundle_contract.py` | Integration bundle manifests must include a ready M018 `integration_policy_manifest`; loose legacy `integration_policy` fields are summary context only | Offline integration evidence only, no hardware, no OSC writes | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_evaluation_integration_bundle_contract.py -q` | legacy policy-only bundles remain blocked; bound manifest passes only offline | revert M019 commit | completed |
| M019-CP02 | Bind M019 to roadmap, completion gates and autopilot state | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/PROJECT_COMPLETION_GATE.md`, `Docs/adr/integration-policy-binding-20260619.md`, `tests/test_wing_mvp_rails.py` | M019 remains pending until integration bundle evidence uses the integration policy contract result | Report-only gate binding | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m019_integration_policy_binding_requires_m018_manifest -q` | roadmap/capability/current-state/report rows | revert M019 commit | completed |

## Validation Contract

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_evaluation_integration_bundle_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_integration_policy_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m019_integration_policy_binding_requires_m018_manifest -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Acceptance Criteria

- Integration bundle manifests consume `integration_policy_manifest` from M018
  before reporting offline integration promotion readiness.
- Legacy `integration_policy` fields without the M018 contract manifest remain
  visible as summary context but cannot satisfy approved-integration-policy
  requirements.
- Integration-policy contract blockers are surfaced with the
  `integration_policy:` child prefix.
- Bound integration policy manifests still cannot grant live authority, close
  readiness, contact WING/backend, send live writes or become hardware proof.
