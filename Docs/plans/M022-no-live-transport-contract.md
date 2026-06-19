# Plan M022: No-live-transport contract

## Checkpoints

| Checkpoint | Scope | Files | Evidence requirement | Safety boundary | Validation | Exit condition | Rollback | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M022-CP01 | Add fail-closed no-live-transport contract | `backend/no_live_transport_contract.py`, `tests/test_no_live_transport_contract.py` | Offline integration candidates must prove artifact-only execution, no live transport and zero live runtime observations | Offline transport evidence only, no hardware, no OSC writes | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_no_live_transport_contract.py -q` | contract result keeps `live_authority_granted: false` and `readiness_closure_allowed: false` | revert M022 commit | completed |
| M022-CP02 | Bind M022 to roadmap and completion gates | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/PROJECT_COMPLETION_GATE.md`, `Docs/adr/no-live-transport-contract-20260619.md`, `tests/test_wing_mvp_rails.py` | M022 remains pending until no-live-transport evidence is complete and offline-only | Report-only gate binding | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m022_no_live_transport_contract_stays_offline_and_artifact_only -q` | roadmap/capability/current-state/report rows | revert M022 commit | completed |

## Validation Contract

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_no_live_transport_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m022_no_live_transport_contract_stays_offline_and_artifact_only -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Acceptance Criteria

- No-live-transport manifests require offline-only controls, evidence artifacts,
  zero runtime transport observations and an empty forbidden-transport token
  list.
- Missing controls, missing evidence, live runtime observations and forbidden
  transport tokens fail closed.
- No-live-transport evidence cannot grant live authority, close readiness,
  contact WING/backend, send live writes or become hardware proof.
