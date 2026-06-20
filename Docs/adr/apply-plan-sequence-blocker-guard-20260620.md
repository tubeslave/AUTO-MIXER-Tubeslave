# ADR: Apply Plan Sequence Blocker Guard

## Context

M037-M046 make the Task Queue and Task Details apply surfaces fail closed while
offline integration bundle blockers are visible. The bottom `Apply Plan` panel
and the Supervised Apply modal `Gate Sequence` still rendered the task's normal
plan steps, so an operator could see `Ready`, `Hold` or pending readback wording
beside controls that are globally blocked by promotion-planning evidence.

## Options Considered

1. Leave the Apply Plan sequence unchanged and rely on nearby blocked notices.
2. Derive a local blocked plan sequence while `integrationBundleApplyBlocked`
   is true.
3. Mutate the task's apply plan input or backend proposal state.

## Decision

Add `applyPlanForIntegrationBundleBlocker`. When the global blocker is absent,
the helper returns the existing plan unchanged. When the blocker is visible, it
returns a local five-step blocked sequence: snapshot held, gate bundle blocked,
apply blocked, read no contact and undo no live. The Task Queue panel and modal
both render from the derived sequence.

This does not mutate apply plan input, task data, queue proposals, open
Supervised Apply, dispatch supervised apply, contact backend/WING, grant live
authority or close readiness.

## Why This Won

The plan sequence is part of the operator's safety explanation. Showing a
derived blocked sequence keeps the visible gate model consistent with the
blocked buttons, badges, metrics and blocker list without touching backend
safety gates or hardware evidence.

## Rejected Alternatives

Leaving the sequence unchanged would preserve a misleading ready-looking path
inside a globally blocked state. Mutating task or backend plan state was
rejected because this checkpoint is frontend presentation only and cannot claim
backend, hardware or operator proof.

## Implementation Plan

1. Add `applyPlanForIntegrationBundleBlocker` in `TaskQueuePanels.tsx`.
2. Feed the bottom `Apply Plan` panel from `guardedApplyPlan`.
3. Feed the Supervised Apply modal `Gate Sequence` from the same guard.
4. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m047_apply_plan_sequence_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Apply Plan sequence presentation. If it regresses,
revert the derived sequence and M047 evidence/test/docs updates. The rollback
does not alter backend gates, WING transport or accepted hardware evidence.
