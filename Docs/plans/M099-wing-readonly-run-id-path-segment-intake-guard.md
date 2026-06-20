# Plan M099: WING Read-Only Run ID Path Segment Intake Guard

## Thesis

M099 keeps the M001 external WING artifact intake bound to one concrete run
directory under the canonical WING read-only artifact root. A candidate
read-only summary must use a `run_id` that is a single safe path segment, not a
path with separators or parent traversal.

## Checkpoints

| Checkpoint | Objective | Evidence |
| --- | --- | --- |
| M099-CP01 | Add a read-only intake guard for path-like `run_id` values | `backend/wing_readonly_preflight_evidence.py` |
| M099-CP02 | Add focused regression coverage for parent-traversal `run_id` candidates | `tests/test_wing_readonly_preflight_evidence.py` |
| M099-CP03 | Bind the guard to roadmap/state/report rails | `tests/test_wing_mvp_rails.py`, `Docs/ROADMAP.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md` |

## Boundary

No live transport, no WING/backend contact, no OSC write, no proposal apply, no
queue import, no readiness closure and no live authority grant.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_evidence.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m099_wing_readonly_run_id_path_segment_intake_guard_stays_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`

## Rollback

Revert the M099 guard, tests and docs/state entries. This does not touch live
console authority or hardware state.
