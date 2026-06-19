# Plan M013: Integration evaluation bundle contract

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M013-CP01 | Add fail-closed integration bundle contract | `backend/evaluation_integration_bundle_contract.py`, `tests/test_evaluation_integration_bundle_contract.py` | Incomplete replay, fault-recovery, integration-check or policy evidence is blocked | Offline integration only, no hardware, no OSC writes | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_evaluation_integration_bundle_contract.py -q` | contract result keeps `live_authority_granted: false` and `readiness_closure_allowed: false` | revert M013 commit | completed |
| M013-CP02 | Bind integration bundle to roadmap and completion gates | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/adr/integration-evaluation-bundle-contract-20260619.md`, `tests/test_wing_mvp_rails.py` | M013 remains pending until replay, fault-recovery, integration trace and approved integration policy evidence all exist | Report-only gate binding | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m013_integration_evaluation_bundle_contract_stays_offline_and_child_gated -q` | roadmap/capability/current-state/report rows | revert M013 commit | completed |

## Required Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_evaluation_integration_bundle_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py --root .`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Acceptance Criteria

- Integration bundle evidence cannot grant live authority or readiness closure.
- Integration bundle evidence cannot be labeled as hardware evidence.
- Offline replay and simulator fault-recovery child contracts must both pass.
- Analysis trace, offline replay result, fault recovery result, safety kernel
  trace, operator report, reproducibility, cross-scenario consistency and
  no-live-transport checks are all required.
- Approved integration policy requires explicit owner and decision id.
- Child contract blockers must be surfaced with stable prefixes so operators
  can see whether replay, recovery or integration evidence is missing.

## Rollback

Revert the M013 commit. The contract is not wired into live runtime paths and
has no console transport, so rollback is a code/docs/test revert only.
