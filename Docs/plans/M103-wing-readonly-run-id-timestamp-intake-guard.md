# Plan M103: WING Read-Only Run ID Timestamp Intake Guard

## Thesis

M103 keeps the M001 external WING artifact intake bound to runner-generated
capture timestamps before a candidate can reach manual review. A candidate
read-only summary must declare a `run_id` whose timestamp portion is a valid
UTC capture time, not only a string that matches the capture-id pattern.

## Checkpoints

| Checkpoint | Objective | Evidence |
| --- | --- | --- |
| M103-CP01 | Add a read-only intake guard for impossible UTC timestamps in WING read-only `run_id` values | `backend/wing_readonly_preflight_evidence.py` |
| M103-CP02 | Add focused regression coverage for self-consistent impossible-timestamp candidates | `tests/test_wing_readonly_preflight_evidence.py` |
| M103-CP03 | Bind the guard to roadmap/state/report rails | `tests/test_wing_mvp_rails.py`, `Docs/ROADMAP.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md` |

## Boundary

No live transport, no WING/backend contact, no OSC write, no proposal apply, no
queue import, no readiness closure and no live authority grant.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_evidence.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m103_wing_readonly_run_id_timestamp_intake_guard_stays_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`

## Rollback

Revert the M103 guard, tests and docs/state entries. This does not touch live
console authority or hardware state.
