# ADR: Task Details Status Value Blocker Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed presentation hardening that prevents operator surfaces
from implying promotion readiness while the integration bundle blocks planning.

Task Details already adds blocked badges and blocks apply/reject controls while
the integration bundle is blocked. The primary task-status badge still renders
the raw task status, which can make a detail panel look applied or currently
actionable even though apply is blocked.

## Options Considered

1. Leave the Task Details primary task-status badge unchanged.
2. Hide the status badge while the integration bundle is blocked.
3. Keep the status badge visible and override only its local display status.

## Decision

Use option 3. `TaskQueuePanels` now reuses
`taskStatusDisplay(task, integrationBundleApplyBlocked)` for the Task Details
primary task-status badge. While blocked, the badge shows `Blocked` with a red
tone. Otherwise it preserves the raw task status and normal status tone.

This does not mutate task status input, task data, proposal data, backend status
input, backend result input, safety monitor input or readiness input.

## Why This Won

The operator keeps useful task status context while the detail panel makes the
global promotion blocker explicit at the status point. It is local presentation
logic only and does not alter task records, backend contracts or apply paths.

## Rejected Alternatives

Leaving the status badge unchanged keeps a misleading detail-view gap. Hiding
the badge removes useful context and is not required for safety because the
badge can fail closed locally.

## Implementation Plan

- Reuse local task status display derivation in `TaskQueuePanels`.
- Render the Task Details primary task-status badge from the local display
  values.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local Task Details status-value
  derivation.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to Task Details presentation. Rollback removes the local
status-value derivation usage, the M065 rails and the M065 docs/state entries.
M001 remains blocked on real WING read-only evidence either way.
