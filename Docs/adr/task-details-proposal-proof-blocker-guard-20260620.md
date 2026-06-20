# ADR: Task Details Proposal Proof Blocker Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed presentation hardening that prevents operator surfaces
from implying promotion readiness while the integration bundle blocks planning.

Task Details already blocks status, risk, delta, rollback, apply/reject controls
and blocker lists while the integration bundle is blocked. The Proposal Preview
`Proof` row still renders the raw proposal proof text, which can make the detail
panel look like live/supervised proof exists even though no live apply occurred.

## Options Considered

1. Leave the Task Details `Proof` row unchanged.
2. Hide the `Proof` row while the integration bundle is blocked.
3. Keep the row visible and override only its local proof display.

## Decision

Use option 3. `TaskQueuePanels` derives the Task Details `Proof` display through
`taskProposalProofDisplay(proposal, integrationBundleApplyBlocked)`. While
blocked, the proof text shows `No live proof` with danger styling. Otherwise it
preserves the raw proposal proof text and normal styling.

This does not mutate proposal data, task status input, task data, backend status
input, backend result input, safety monitor input or readiness input.

## Why This Won

The operator keeps proof context while the detail panel makes the global
promotion blocker explicit at the proof point. It is local presentation logic
only and does not alter proposal records, task records, backend contracts,
modal confirm guards, apply paths or supervised proof requirements.

## Rejected Alternatives

Leaving the raw proof unchanged keeps a misleading Task Details gap. Hiding the
row removes useful context and is not required for safety because the proof text
can fail closed locally.

## Implementation Plan

- Add local proof display derivation in `TaskQueuePanels`.
- Render Task Details Proposal Preview `Proof` from the local display values.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local Task Details proof derivation.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to Task Details presentation. Rollback removes the local proof
binding, the M072 rails and the M072 docs/state entries. M001 remains blocked
on real WING read-only evidence either way.
