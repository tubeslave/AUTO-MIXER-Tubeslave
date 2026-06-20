# Plan M098: WING Read-Only Source Path Intake Guard

## Thesis

M098 keeps the M001 external WING artifact intake bound to the actual summary
file being reviewed. A candidate read-only summary loaded from disk must have a
`summary_path` field that matches the source file path.

## Checkpoints

| Checkpoint | Objective | Evidence |
| --- | --- | --- |
| M098-CP01 | Add a read-only intake guard for copied or stale source summary files | `backend/wing_readonly_preflight_evidence.py` |
| M098-CP02 | Add focused regression coverage for source-path/declared-summary mismatch | `tests/test_wing_readonly_preflight_evidence.py` |
| M098-CP03 | Bind the guard to roadmap/state/report rails | `tests/test_wing_mvp_rails.py`, `Docs/ROADMAP.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md` |

## Boundary

No live transport, no WING/backend contact, no OSC write, no proposal apply, no
queue import, no readiness closure and no live authority grant.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_evidence.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m098_wing_readonly_source_path_intake_guard_stays_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`

## Rollback

Revert the M098 guard, tests and docs/state entries. This does not touch live
console authority or hardware state.
