# Plan M011: Offline replay evaluation contract

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M011-CP01 | Add fail-closed replay evaluation contract | `backend/offline_replay_evaluation_contract.py`, `tests/test_offline_replay_evaluation_contract.py` | Incomplete soundcheck/studio/concert replay manifests, pending thresholds, live claims and hardware labels are blocked | Offline only, no hardware, no OSC writes | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_offline_replay_evaluation_contract.py -q` | contract result keeps `live_authority_granted: false` and `readiness_closure_allowed: false` | revert M011 commit | completed |
| M011-CP02 | Bind contract to roadmap and completion gates | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/adr/offline-replay-evaluation-contract-20260619.md`, `tests/test_wing_mvp_rails.py` | M011 remains pending until approved thresholds and complete scenario evidence exist | Report-only gate binding | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m011_offline_replay_evaluation_contract_stays_offline_and_threshold_gated -q` | roadmap/capability/current-state/report rows | revert M011 commit | completed |

## Required Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_offline_replay_evaluation_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py --root .`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Acceptance Criteria

- Offline replay evidence cannot grant live authority or readiness closure.
- Offline replay evidence cannot be labeled as hardware evidence.
- Soundcheck, studio and concert scenarios are all required.
- Offline, replay and fault-injection evidence are all required per scenario.
- Objective metrics, reference comparison, repeatability, blind listening,
  expert assessment, clipping/overload, masking/intelligibility, latency,
  oscillation and recovery/failure quality axes are all required.
- Approved threshold policy requires explicit owner and decision id.

## Rollback

Revert the M011 commit. The contract is not wired into live runtime paths and
has no console transport, so rollback is a code/docs/test revert only.

