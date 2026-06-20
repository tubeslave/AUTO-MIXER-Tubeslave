# ADR: Supervised Apply Old Value Blocker Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed presentation hardening that prevents operator surfaces
from implying promotion readiness while the integration bundle blocks planning.

Supervised Apply already blocks confirm controls, gate rows, backend results,
readiness, parameter, new value and rollback while the integration bundle is
blocked. The Proposal Target `Old value` row still renders `Current console
snapshot`, which can make the modal look like a real backend/console readback is
available even though no backend contact or live readback is allowed.

## Options Considered

1. Leave the Supervised Apply `Old value` row unchanged.
2. Hide the `Old value` row while the integration bundle is blocked.
3. Keep the row visible and override only its local old-value display.

## Decision

Use option 3. `TaskQueuePanels` derives the Supervised Apply `Old value` display
through `supervisedProposalOldValueDisplay(integrationBundleApplyBlocked)`.
While blocked, the old value text shows `No readback` with danger styling.
Otherwise it preserves `Current console snapshot` and normal muted styling.

This does not mutate proposal data, task status input, task data, backend status
input, backend result input, safety monitor input or readiness input.

## Why This Won

The operator keeps the Proposal Target layout while the modal makes the missing
readback explicit. It is local presentation logic only and does not alter
proposal records, task records, backend contracts, modal confirm guards, apply
paths or supervised proof requirements.

## Rejected Alternatives

Leaving `Current console snapshot` unchanged keeps a misleading Supervised Apply
gap. Hiding the row removes useful context and is not required for safety
because the old-value text can fail closed locally.

## Implementation Plan

- Add local old-value display derivation in `TaskQueuePanels`.
- Render Supervised Apply Proposal Target `Old value` from the local display
  values.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local Supervised Apply old-value
  derivation.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to Supervised Apply presentation. Rollback removes the local
old-value binding, the M075 rails and the M075 docs/state entries. M001 remains
blocked on real WING read-only evidence either way.
