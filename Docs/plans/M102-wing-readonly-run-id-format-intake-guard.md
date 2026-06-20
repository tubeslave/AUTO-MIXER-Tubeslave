# Plan M102: WING Read-Only Run ID Format Intake Guard

## Thesis

M102 keeps the M001 external WING artifact intake bound to runner-generated
capture directories before a candidate can reach manual review. A candidate
read-only summary must declare a `run_id` in the canonical UTC runner format:
`YYYYMMDDTHHMMSSZ_wing_readonly_preflight`.

## Checkpoints

| Checkpoint | Objective | Evidence |
| --- | --- | --- |
| M102-CP01 | Add a read-only intake guard for non-canonical WING read-only `run_id` values | `backend/wing_readonly_preflight_evidence.py` |
| M102-CP02 | Add focused regression coverage for self-consistent manual-run-id candidates | `tests/test_wing_readonly_preflight_evidence.py` |
| M102-CP03 | Bind the guard to roadmap/state/report rails | `tests/test_wing_mvp_rails.py`, `Docs/ROADMAP.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md` |
| M102-CP04 | Keep downstream review-decision multi-candidate fixtures in canonical runner format | `tests/test_wing_readonly_preflight_review_tool.py` |

## Boundary

No live transport, no WING/backend contact, no OSC write, no proposal apply, no
queue import, no readiness closure and no live authority grant.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_evidence.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_review_tool.py::test_review_decision_template_requires_explicit_source_for_multiple_candidates -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m102_wing_readonly_run_id_format_intake_guard_stays_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`

## Rollback

Revert the M102 guard, tests and docs/state entries. This does not touch live
console authority or hardware state.
