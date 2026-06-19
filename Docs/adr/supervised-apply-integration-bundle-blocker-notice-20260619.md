# ADR: Supervised Apply Integration Bundle Blocker Notice

## Context

M029-M035 made Director, Agent Suggestions and Task Queue surfaces show offline
integration bundle blockers as read-only operator evidence. The Supervised
Apply modal is the closest frontend surface to a supervised apply request, so a
blocked offline integration bundle also needs to be visible there before
proposal review.

M001 still waits for a real WING read-only artifact and operator review. This
decision is frontend/read-only Supervised Apply display only.

## Options Considered

1. Leave the Supervised Apply modal unchanged and rely on Task Queue notices.
2. Show a blocker notice inside the Supervised Apply modal while keeping the
   existing apply gate labels and controls governed by their current status.
3. Disable or hide every modal action whenever the offline integration bundle
   is blocked.

## Decision

Show a read-only Supervised Apply notice:
`Promotion planning blocked. Supervised apply remains blocked` when Director
runtime blocked-reason messages and a modal task are visible.

This does not queue proposals, bypass apply gates, contact backend/WING, send
transport, grant readiness closure or grant live authority.

## Why This Won

The operator should see the blocker at the point closest to a supervised apply
request. A local notice keeps modal review truthful while preserving the
existing apply-gate explanation and the audit value of proposal details.

## Rejected Alternatives

Leaving the modal unchanged was rejected because it requires the operator to
remember the Task Queue or Director blocker while reviewing the apply modal.

Hiding or disabling every modal action was rejected because live eligibility
must be governed by explicit proposal, proof and hardware gates rather than a
broad UI disappearance.

## Implementation Plan

- Add an `integrationBundleBlockerNotice` prop to `SupervisedApplyModal`.
- Derive `supervisedApplyBlockerNotice` from
  `directorBlockedReasonMessages(runtime.blockedReasons)` when a modal task is
  active.
- Render the notice before the modal apply review grid.
- Extend
  `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs` to assert
  the notice wiring in the active Task Queue panel source.
- Update roadmap, completion gate, capability matrix, current state and
  autopilot evidence.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m036_supervised_apply_modal_shows_integration_bundle_blocker_notice -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: the notice could be read as replacing the existing apply gate. The
wording ties it to promotion planning and the modal still renders its existing
gate sequence and backend-result labels.

Rollback: revert the M036 Supervised Apply modal/test/docs/state changes. The
M026-M035 readiness, Director, Agent Suggestions and Task Queue surfaces remain
available.
