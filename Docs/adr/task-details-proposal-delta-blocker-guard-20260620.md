# ADR: Task Details Proposal Delta Blocker Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed presentation hardening that prevents operator surfaces
from implying promotion readiness while the integration bundle blocks planning.

Task Details already blocks apply/reject controls and shows blocked status and
proposal risk while the integration bundle is blocked. The Proposal Preview
`Delta` row still renders the raw proposal delta, which can make the detail view
look ready to apply even though apply is blocked.

## Options Considered

1. Leave the Task Details proposal delta unchanged.
2. Hide the `Delta` row while the integration bundle is blocked.
3. Keep the row visible and override only its local proposal-delta display.

## Decision

Use option 3. `TaskQueuePanels` derives local Task Details proposal-delta display
values through `taskProposalDeltaDisplay(proposal, integrationBundleApplyBlocked)`.
While blocked, the delta text shows `Blocked` with danger styling. Otherwise it
preserves the raw proposal delta and normal move styling.

This does not mutate proposal data, task status input, task data, backend status
input, backend result input, safety monitor input or readiness input.

## Why This Won

The operator keeps proposal context while the detail view makes the global
promotion blocker explicit at the proposed-delta point. It is local presentation
logic only and does not alter proposal records, task records, backend contracts
or apply paths.

## Rejected Alternatives

Leaving the proposal delta unchanged keeps a misleading Task Details gap. Hiding
the row removes useful context and is not required for safety because the delta
text can fail closed locally.

## Implementation Plan

- Add local proposal-delta display derivation in `TaskQueuePanels`.
- Render the Task Details Proposal Preview delta from the local display values.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local Task Details proposal-delta
  derivation.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to Task Details presentation. Rollback removes the local
proposal-delta derivation, the M068 rails and the M068 docs/state entries. M001
remains blocked on real WING read-only evidence either way.
