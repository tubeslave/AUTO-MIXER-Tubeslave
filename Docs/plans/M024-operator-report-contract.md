# Plan M024: Operator report contract

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M024-CP01 | Define operator-report evidence contract | `backend/operator_report_contract.py`, `tests/test_operator_report_contract.py` | Operator-facing integration reports fail closed unless required sections, report artifact bindings, authority-denial flags, visible blockers, next action and audit fields are present | Offline only, no WING/backend contact, no live writes | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_operator_report_contract.py -q` | pending validation | revert M024 code/test commit | completed |
| M024-CP02 | Bind roadmap, rails and evidence state | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/PROJECT_COMPLETION_GATE.md`, `tests/test_wing_mvp_rails.py`, `ops/autopilot/*` | Roadmap, current state and reports say operator-report evidence is offline/review-bound while live authority remains denied | Report/test only | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m024_operator_report_contract_stays_offline_and_review_bound -q` | pending validation | revert M024 docs/state commit | completed |

## Required Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_operator_report_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m024_operator_report_contract_stays_offline_and_review_bound -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Acceptance Criteria

- Incomplete operator reports are blocked with missing section, binding,
  authority-denial, visible-blocker, next-action and audit blockers.
- Complete offline operator reports can support offline review use.
- Operator-report evidence cannot grant live authority, close readiness,
  contact WING/backend, claim hardware evidence or replace real WING proof.
- M001 remains `WAITING_FOR_EXTERNAL_VALIDATION` for real WING read-only
  artifact and operator review.

## Rollback

Revert the M024 commit. No hardware, WING session, backend session or live write
state is created by this milestone.
