# ADR: Safety Monitor Goal Readiness Blocker Guard

## Context

M056 made the Safety Monitor `Policy` row fail closed while offline
integration bundle blockers are visible. The same Safety Monitor panel still
rendered `Goal readiness` as a percentage, which can look like partial
readiness even while promotion planning is globally blocked.

## Options Considered

1. Leave Safety Monitor `Goal readiness` unchanged because other panels already
   show the blocker.
2. Derive a local blocked goal-readiness presentation from
   `integrationBundleApplyBlocked`, while preserving the raw readiness score
   when the blocker is absent.
3. Hide the Safety Monitor panel while promotion planning is blocked.

## Decision

Add local Safety Monitor goal-readiness presentation state derived from
`integrationBundleApplyBlocked`. When the global blocker is visible, the panel
shows `Blocked by integration bundle` with danger styling. When the blocker is
absent, the existing readiness score and tone remain unchanged.

This does not mutate readiness input, safety monitor input, backend status
input, backend result input, task data, queue proposals, open Supervised Apply,
dispatch supervised apply, contact backend/WING, grant live authority or close
readiness.

## Why This Won

Safety Monitor is the operator's compact safety summary. A percentage in the
Goal readiness row can imply progress toward readiness even when the global
integration-bundle blocker says promotion planning remains blocked. A local
presentation guard keeps the summary aligned with the global blocker without
altering backend readiness evidence.

## Rejected Alternatives

Leaving the row unchanged preserved ambiguity on a safety surface. Hiding the
panel was rejected because the operator still needs emergency, trace and policy
evidence while reviewing blocked promotion-planning state.

## Implementation Plan

1. Add `goalReadinessStatus` in `SafetyMonitorPanel`.
2. Render `Blocked by integration bundle` only while
   `integrationBundleApplyBlocked` is true.
3. Preserve existing readiness score display and tone when the blocker is
   absent.
4. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m057_safety_monitor_goal_readiness_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Safety Monitor row presentation. If it regresses,
revert the local goal-readiness presentation guard and M057 evidence/test/docs
updates. The rollback does not alter backend gates, WING transport or accepted
hardware evidence.
