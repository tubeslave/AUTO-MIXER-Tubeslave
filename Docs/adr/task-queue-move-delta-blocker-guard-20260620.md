# ADR: Task Queue Move Delta Blocker Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed presentation hardening that prevents operator surfaces
from implying promotion readiness while the integration bundle blocks planning.

Task Queue already blocks row actions and shows blocked statuses while the
integration bundle is blocked. The `Target / Move` column still renders the raw
proposal delta, which can make a row look ready to apply even though apply is
blocked.

## Options Considered

1. Leave the Task Queue move delta unchanged.
2. Hide the `Target / Move` column while the integration bundle is blocked.
3. Keep the column visible and override only its local move display.

## Decision

Use option 3. `TaskQueuePanels` derives local Task Queue move display values
through `taskMoveDisplay(task, integrationBundleApplyBlocked)`. While blocked,
the move text shows `Blocked` with danger styling. Otherwise it preserves the
raw proposal delta and normal move styling.

This does not mutate proposal data, task status input, task data, backend status
input, backend result input, safety monitor input or readiness input.

## Why This Won

The operator keeps target context while the queue makes the global promotion
blocker explicit at the proposed-move point. It is local presentation logic only
and does not alter proposal records, task records, backend contracts or apply
paths.

## Rejected Alternatives

Leaving the move delta unchanged keeps a misleading queue gap. Hiding the column
removes useful target context and is not required for safety because the move
text can fail closed locally.

## Implementation Plan

- Add local move-delta display derivation in `TaskQueuePanels`.
- Render the Task Queue `Target / Move` delta from the local display values.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local Task Queue move-delta derivation.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to Task Queue presentation. Rollback removes the local
move-delta derivation, the M067 rails and the M067 docs/state entries. M001
remains blocked on real WING read-only evidence either way.
