# ADR: Task Details Proposal Risk Blocker Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed presentation hardening that prevents operator surfaces
from implying promotion readiness while the integration bundle blocks planning.

Task Details already blocks apply/reject controls and now shows blocked task
status while the integration bundle is blocked. The `Proposal Preview` risk badge
still renders the raw proposal risk, which can make a proposal look safe or
currently actionable even though apply is blocked.

## Options Considered

1. Leave the Task Details proposal risk badge unchanged.
2. Hide the proposal preview while the integration bundle is blocked.
3. Keep the proposal preview visible and override only its local risk display.

## Decision

Use option 3. `TaskQueuePanels` derives local Task Details proposal-risk display
values through `taskProposalRiskDisplay(proposal, integrationBundleApplyBlocked)`.
While blocked, the badge shows `Blocked` with a red tone. Otherwise it preserves
the raw proposal risk and normal risk tone.

This does not mutate proposal data, task status input, task data, backend status
input, backend result input, safety monitor input or readiness input.

## Why This Won

The operator keeps proposal context while the detail panel makes the global
promotion blocker explicit at the risk point. It is local presentation logic only
and does not alter proposal records, task records, backend contracts or apply
paths.

## Rejected Alternatives

Leaving the risk badge unchanged keeps a misleading detail-view gap. Hiding the
proposal preview removes useful review context and is not required for safety
because the badge can fail closed locally.

## Implementation Plan

- Add local proposal-risk display derivation in `TaskQueuePanels`.
- Render the Task Details proposal risk badge from the local display values.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local Task Details proposal-risk
  derivation.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to Task Details presentation. Rollback removes the local
proposal-risk derivation, the M066 rails and the M066 docs/state entries. M001
remains blocked on real WING read-only evidence either way.
