# ADR: Replay Evidence Decision Trace Status Blocker Guard

## Context

M001 is still blocked on external WING/operator validation. The project can
continue only with offline, fail-closed presentation hardening that makes
operator surfaces consistent about blocked promotion planning.

Replay Evidence shows the latest decision trace summary. When the offline
integration bundle blocks promotion planning, that trace surface needs a local
blocked status so it cannot look usable or verified in isolation.

## Options Considered

1. Leave Replay Evidence unchanged.
2. Hide the trace summary while the integration bundle is blocked.
3. Keep the trace summary visible and add a local blocked status badge.

## Decision

Use option 3. `ReplayEvidencePanel` receives
`integrationBundleApplyBlocked` and derives local `traceStatusLabel` and
`traceStatusTone`. While blocked, the Trace row shows `bundle blocked` with a
red tone. Otherwise it preserves the trace `blocked` or `proofStatus` status.

This does not mutate decision trace input, replay input, task data, backend
status input, backend result input, safety monitor input or readiness input.

## Why This Won

The operator keeps useful offline trace context while the UI makes the global
promotion blocker explicit at the point of reading the trace. It is the smallest
visible change and does not alter backend contracts, trace records or apply
paths.

## Rejected Alternatives

Leaving the surface unchanged keeps a misleading gap between global blockers and
trace status. Hiding traces removes useful offline evidence and makes diagnosis
harder without improving live safety.

## Implementation Plan

- Add optional `integrationBundleApplyBlocked` to `ReplayEvidencePanel`.
- Derive local trace status values from the blocker before rendering.
- Pass the existing Task Queue integration-bundle blocker flag into
  `ReplayEvidencePanel`.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local status derivation and parent prop.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to the Replay Evidence presentation. Rollback removes the local
status derivation, the prop pass-through, the M059 rails and the M059 docs/state
entries. M001 remains blocked on real WING read-only evidence either way.
