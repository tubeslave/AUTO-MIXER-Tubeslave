# Plan M104: WING Read-Only Capture Time Order Intake Guard

## Thesis

M104 keeps the M001 external WING artifact intake bound to a coherent capture
timeline before a candidate can reach manual review. A candidate read-only
summary must not claim a `captured_at` time earlier than the UTC capture
timestamp encoded in its runner-generated `run_id`.

## Checkpoints

| Checkpoint | Objective | Evidence |
| --- | --- | --- |
| M104-CP01 | Add a read-only intake guard for `captured_at` values that precede the `run_id` UTC timestamp | `backend/wing_readonly_preflight_evidence.py` |
| M104-CP02 | Add focused regression coverage for stale or cross-run capture-time ordering and keep review-template fixtures coherent with the new event-time lower bound | `tests/test_wing_readonly_preflight_evidence.py`, `tests/test_wing_readonly_preflight_review_tool.py` |
| M104-CP03 | Bind the guard to roadmap/state/report rails | `tests/test_wing_mvp_rails.py`, `Docs/ROADMAP.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md` |

## Boundary

No live transport, no WING/backend contact, no OSC write, no proposal apply, no
queue import, no readiness closure and no live authority grant.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_evidence.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_review_tool.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m104_wing_readonly_capture_time_order_intake_guard_stays_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`

## Rollback

Revert the M104 guard, tests and docs/state entries. This does not touch live
console authority or hardware state.
