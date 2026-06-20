# ADR: Replay Evidence Channel Identity Blocker Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed UI hardening that prevents replay evidence surfaces
from implying promotion readiness while integration-bundle blockers are
visible.

Replay Evidence already blocks its trace status/detail, latest-run status/id,
policy row, channel status, report path and start action. The channel identity
text still renders raw channel id/name evidence, which can make stale channel
evidence look current while replay start is blocked.

## Options Considered

1. Leave the channel identity unchanged and rely on the blocked status badge.
2. Hide the channel row while blocked.
3. Keep the row visible and override only the local channel-identity display.

## Decision

Use option 3. `TaskQueuePanels` derives `channelEvidenceLabel` and
`channelEvidenceClassName` from `integrationBundleApplyBlocked`. While blocked,
the row shows `No channel proof` with danger styling. Otherwise it preserves the
raw channel id/name or `No channel evidence`.

This does not mutate channel evidence input, replay run input, decision trace
input, task data, backend status input, backend result input, safety monitor
input or readiness input.

## Why This Won

The operator keeps the Replay Evidence layout while the missing channel-proof
state is explicit. The change is local presentation logic only and does not
alter replay artifacts, backend contracts, WING preflight behavior, or
supervised apply paths.

## Rejected Alternatives

Leaving raw channel identity visible keeps a stale-evidence gap. Hiding the row
removes useful context and is not required because the value can fail closed
locally.

## Implementation Plan

- Add local channel-identity display derivation in `TaskQueuePanels`.
- Render Replay Evidence channel identity from the local display values.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local channel-identity derivation and the
  absence of direct raw channel-identity rendering.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to Replay Evidence presentation while global blockers are
visible. Rollback removes the local channel-identity binding, the M079 rails and
the M079 docs/state entries. M001 remains blocked on real WING read-only
evidence either way.
