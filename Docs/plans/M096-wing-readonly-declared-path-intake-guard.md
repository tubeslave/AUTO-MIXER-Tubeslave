# Plan M096: WING Read-Only Declared Path Intake Guard

## Thesis

M096 keeps M001 focused on the real WING read-only artifact by rejecting
candidate summaries whose declared `summary_path` or `events_path` does not
match `artifact_root/<run_id>/summary.json|events.jsonl`.

## Checkpoints

| Checkpoint | Objective | Evidence |
| --- | --- | --- |
| M096-CP01 | Add a read-only intake guard for declared summary/events path shape | `backend/wing_readonly_preflight_evidence.py` |
| M096-CP02 | Add focused regression coverage for direct candidate payload mismatch | `tests/test_wing_readonly_preflight_evidence.py` |
| M096-CP03 | Bind the guard to roadmap/state/report rails | `tests/test_wing_mvp_rails.py`, `Docs/ROADMAP.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md` |

## Boundary

No live transport, no WING/backend contact, no OSC write, no proposal apply, no
queue import, no readiness closure and no live authority grant.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_evidence.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_review_decision.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m096_wing_readonly_declared_path_intake_guard_stays_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`

## Rollback

Revert the M096 guard, tests and docs/state entries. This does not touch live
console authority or hardware state.
