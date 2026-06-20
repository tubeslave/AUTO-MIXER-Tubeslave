# Plan M100: WING Read-Only Artifact Root Parent Traversal Intake Guard

## Thesis

M100 keeps the M001 external WING artifact intake bound to the canonical
artifact root before a candidate can reach manual review. A candidate read-only
summary must not declare an `artifact_root` containing parent traversal.

## Checkpoints

| Checkpoint | Objective | Evidence |
| --- | --- | --- |
| M100-CP01 | Add a read-only intake guard for parent traversal in `artifact_root` | `backend/wing_readonly_preflight_evidence.py` |
| M100-CP02 | Add focused regression coverage for self-consistent traversal-root candidates | `tests/test_wing_readonly_preflight_evidence.py` |
| M100-CP03 | Bind the guard to roadmap/state/report rails | `tests/test_wing_mvp_rails.py`, `Docs/ROADMAP.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md` |

## Boundary

No live transport, no WING/backend contact, no OSC write, no proposal apply, no
queue import, no readiness closure and no live authority grant.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_evidence.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m100_wing_readonly_artifact_root_parent_traversal_intake_guard_stays_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`

## Rollback

Revert the M100 guard, tests and docs/state entries. This does not touch live
console authority or hardware state.
