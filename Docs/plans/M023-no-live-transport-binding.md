# Plan M023: No-live-transport binding

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M023-CP01 | Bind no-live-transport manifest into integration bundle | `backend/evaluation_integration_bundle_contract.py`, `tests/test_evaluation_integration_bundle_contract.py` | Offline integration promotion consumes the M022 no-live-transport contract result instead of loose `no_live_transport` integration checks | Offline only, no WING/backend contact, no live writes | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_evaluation_integration_bundle_contract.py -q` | pending validation | revert M023 code/test commit | completed |
| M023-CP02 | Bind docs, rails and autopilot evidence | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/PROJECT_COMPLETION_GATE.md`, `tests/test_wing_mvp_rails.py`, `ops/autopilot/*` | Roadmap, current state and reports say integration promotion requires bound no-live-transport manifest while live authority remains denied | Report/test only | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m023_no_live_transport_binding_requires_m022_manifest -q` | pending validation | revert M023 docs/state commit | completed |

## Required Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_evaluation_integration_bundle_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_no_live_transport_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m023_no_live_transport_binding_requires_m022_manifest -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Acceptance Criteria

- A complete integration bundle is blocked when it has only a loose `no_live_transport` check and no M022 manifest.
- The integration bundle exposes prefixed `no_live_transport:*` blockers from the child contract.
- A bound M022 manifest removes no-live-transport blockers without granting live authority, closing readiness, contacting WING/backend or claiming hardware evidence.
- M001 remains `WAITING_FOR_EXTERNAL_VALIDATION` for real WING read-only artifact and operator review.

## Rollback

Revert the M023 commit. The rollback returns integration bundles to the previous
standalone M022 contract state. No hardware, WING session or live write state is
created by this milestone.
