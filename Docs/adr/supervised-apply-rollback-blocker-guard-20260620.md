# ADR: Supervised Apply Rollback Blocker Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed presentation hardening that prevents operator surfaces
from implying promotion readiness while the integration bundle blocks planning.

Supervised Apply already blocks confirm, hides matched backend results and shows
blocked gate rows while the integration bundle is blocked. The Proposal Target
`Rollback` row still renders the raw proposal rollback text, which can make the
modal look like a live rollback path exists even though no live apply occurred.

## Options Considered

1. Leave the Supervised Apply `Rollback` row unchanged.
2. Hide the `Rollback` row while the integration bundle is blocked.
3. Keep the row visible and override only its local rollback display.

## Decision

Use option 3. `TaskQueuePanels` derives the Supervised Apply `Rollback` display
through `taskProposalRollbackDisplay(proposal, integrationBundleApplyBlocked)`.
While blocked, the rollback text shows `No live apply` with danger styling.
Otherwise it preserves the raw proposal rollback text and normal styling.

This does not mutate proposal data, task status input, task data, backend status
input, backend result input, safety monitor input or readiness input.

## Why This Won

The operator keeps rollback context while the modal makes the global promotion
blocker explicit at the rollback point. It is local presentation logic only and
does not alter proposal records, task records, backend contracts, modal confirm
guards, apply paths or rollback proof requirements.

## Rejected Alternatives

Leaving the raw rollback unchanged keeps a misleading Supervised Apply gap.
Hiding the row removes useful context and is not required for safety because the
rollback text can fail closed locally.

## Implementation Plan

- Add local rollback display derivation in `TaskQueuePanels`.
- Render Supervised Apply Proposal Target `Rollback` from the local display
  values.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local Supervised Apply rollback
  derivation.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to Supervised Apply modal presentation. Rollback removes the
local rollback binding, the M070 rails and the M070 docs/state entries. M001
remains blocked on real WING read-only evidence either way.
