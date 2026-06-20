# ADR: Replay Evidence Start Action Blocker Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed UI hardening that prevents operator surfaces from
implying promotion readiness or initiating backend work while integration-bundle
blockers are visible.

Replay Evidence status rows already show blocked integration-bundle state.
However, the `Start Offline` action still binds directly to `onStartReplay`.
The current frontend action path can request a replay run through the Product
Layer websocket. While promotion planning is blocked, that button should not
start a backend replay request from the blocked evidence surface.

## Options Considered

1. Leave `Start Offline` enabled and rely on backend replay policy.
2. Hide the Replay Evidence action while blocked.
3. Keep the action visible, disabled and labeled as blocked, with a local
   click-handler return before `onStartReplay`.

## Decision

Use option 3. `TaskQueuePanels` derives `replayStartBlocked`,
`replayStartLabel`, `replayStartButtonClassName` and `handleStartReplay` from
`integrationBundleApplyBlocked`. While blocked, the button shows
`Replay Blocked`, is disabled, and the click handler returns before
`onStartReplay`. Otherwise the existing `Start Offline` behavior is preserved.

This does not mutate replay run input, decision trace input, channel evidence
input, task data, backend status input, backend result input, safety monitor
input or readiness input.

## Why This Won

The operator can still see that replay evidence exists, but the blocked surface
cannot initiate backend replay work. The change is local frontend guard logic
and does not change replay contracts, backend policy, WING preflight behavior,
or supervised apply paths.

## Rejected Alternatives

Leaving the action enabled lets a blocked promotion surface still initiate a
backend replay request. Hiding the action removes useful state from the panel
and makes the blocked reason less visible than a disabled `Replay Blocked`
control.

## Implementation Plan

- Add local Replay Evidence start-action blocked derivation in
  `TaskQueuePanels`.
- Render the action from the derived label, class and disabled state.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local replay-start guard and the absence
  of direct `onClick={onStartReplay}`.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to the Replay Evidence panel action while global blockers are
visible. Rollback removes the local replay-start binding, the M076 rails and
the M076 docs/state entries. M001 remains blocked on real WING read-only
evidence either way.
