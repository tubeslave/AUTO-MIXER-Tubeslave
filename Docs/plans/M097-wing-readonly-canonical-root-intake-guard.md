# Plan M097: WING Read-Only Canonical Root Intake Guard

## Thesis

M097 keeps the M001 external WING artifact path unambiguous by rejecting
candidate read-only summaries whose `artifact_root` is not the canonical WING
read-only capture root, `artifacts/wing_readonly_preflight`.

## Checkpoints

| Checkpoint | Objective | Evidence |
| --- | --- | --- |
| M097-CP01 | Add a read-only intake guard for non-canonical WING artifact roots | `backend/wing_readonly_preflight_evidence.py` |
| M097-CP02 | Add focused regression coverage for non-canonical direct candidate roots | `tests/test_wing_readonly_preflight_evidence.py` |
| M097-CP03 | Bind the guard to roadmap/state/report rails | `tests/test_wing_mvp_rails.py`, `Docs/ROADMAP.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md` |

## Boundary

No live transport, no WING/backend contact, no OSC write, no proposal apply, no
queue import, no readiness closure and no live authority grant.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_evidence.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m097_wing_readonly_canonical_root_intake_guard_stays_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`

## Rollback

Revert the M097 guard, tests and docs/state entries. This does not touch live
console authority or hardware state.
