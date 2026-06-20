# ADR: Safety Monitor Policy Blocker Guard

## Context

The Task Queue workspace now shows integration-bundle blockers across proposal
actions, Task Details, Operator Decision and Supervised Apply. The lower Safety
Monitor panel still rendered the raw `safety.liveWritePolicy` row directly,
which can look like a normal green supervised policy while promotion planning
is globally blocked.

## Options Considered

1. Leave Safety Monitor `Policy` unchanged because other panels already show
   the blocker.
2. Derive a local blocked Safety Monitor policy presentation from
   `integrationBundleApplyBlocked`, while preserving the raw safety policy when
   the blocker is absent.
3. Hide the Safety Monitor panel while promotion planning is blocked.

## Decision

Add local Safety Monitor policy presentation state derived from
`integrationBundleApplyBlocked`. When the global blocker is visible, the panel
shows `Integration bundle blocked` with danger styling. When the blocker is
absent, the existing `safety.liveWritePolicy` behavior remains unchanged.

This does not mutate safety monitor input, backend status input, backend result
input, task data, queue proposals, open Supervised Apply, dispatch supervised
apply, contact backend/WING, grant live authority or close readiness.

## Why This Won

Safety Monitor is the operator's compact safety summary. Keeping a
ready-looking policy row on that panel while every apply surface is blocked
creates a mixed signal. A local presentation guard keeps the safety summary
aligned with the global blocker without altering backend safety evidence.

## Rejected Alternatives

Leaving the row unchanged preserved ambiguity on a safety surface. Hiding the
panel was rejected because the operator still needs emergency, trace and
readiness evidence while reviewing blocked promotion-planning state.

## Implementation Plan

1. Add `policyStatus` in `SafetyMonitorPanel`.
2. Render `Integration bundle blocked` only while
   `integrationBundleApplyBlocked` is true.
3. Preserve existing `safety.liveWritePolicy` display and tone when the blocker
   is absent.
4. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m056_safety_monitor_policy_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Safety Monitor row presentation. If it regresses,
revert the local policy presentation guard and M056 evidence/test/docs updates.
The rollback does not alter backend gates, WING transport or accepted hardware
evidence.
