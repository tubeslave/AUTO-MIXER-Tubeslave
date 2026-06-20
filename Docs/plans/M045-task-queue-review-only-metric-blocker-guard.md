# Plan M045: Task Queue review-only metric blocker guard

## Goal

Make the Task Queue `Review Only` metric count every row when offline
integration bundle blockers make every apply path review-only.

## Scope

- `frontend/src/components/prototype/TaskQueuePanels.tsx`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `tests/test_wing_mvp_rails.py`
- roadmap, capability, completion-gate, ADR and autopilot evidence updates

## Out of Scope

- No backend contact.
- No WING contact.
- No OSC or websocket transport.
- No row data mutation.
- No queue mutation.
- No supervised apply dispatch.
- No readiness closure.
- No live authority grant.
- No Full Autonomous readiness claim.

## Checkpoints

| Checkpoint | Objective | Evidence | Status |
| --- | --- | --- | --- |
| M045-CP01 | Make the `Review Only` metric fail closed while integration bundle apply is blocked | `TaskQueuePanels.tsx` metric count guard | planned |
| M045-CP02 | Add contract and rail coverage for the blocked review-only metric count | frontend contract test plus `test_wing_mvp_rails.py` | planned |

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m045_task_queue_review_only_metric_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Rollback

Revert the Task Queue review-only metric count guard, M045 frontend contract
test, M045 rail, roadmap row, ADR, completion-gate paragraph and autopilot
evidence updates. Rollback is local and does not touch backend safety gates or
hardware evidence.
