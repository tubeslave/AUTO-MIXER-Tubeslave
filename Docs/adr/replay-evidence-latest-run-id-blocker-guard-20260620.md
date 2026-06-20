# ADR: Replay Evidence Latest Run Id Blocker Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed UI hardening that prevents replay evidence surfaces
from implying promotion readiness while integration-bundle blockers are
visible.

Replay Evidence already blocks its latest-run status badge, policy row, channel
row, report path and start action. The `Latest run` text still renders the raw
`runId`, which can make stale replay evidence look current while replay start is
blocked.

## Options Considered

1. Leave the latest-run id unchanged and rely on the blocked badge.
2. Hide the `Latest run` row while blocked.
3. Keep the row visible and override only the local latest-run id display.

## Decision

Use option 3. `TaskQueuePanels` derives `latestRunIdLabel` and
`latestRunIdClassName` from `integrationBundleApplyBlocked`. While blocked, the
row shows `No replay start` with danger styling. Otherwise it preserves the raw
run id or `No replay yet`.

This does not mutate replay run input, decision trace input, channel evidence
input, task data, backend status input, backend result input, safety monitor
input or readiness input.

## Why This Won

The operator keeps the Replay Evidence layout while the missing replay-start
state is explicit. The change is local presentation logic only and does not
alter replay artifacts, backend contracts, WING preflight behavior, or
supervised apply paths.

## Rejected Alternatives

Leaving the raw run id visible keeps a stale-evidence gap. Hiding the row
removes useful context and is not required because the value can fail closed
locally.

## Implementation Plan

- Add local latest-run id display derivation in `TaskQueuePanels`.
- Render Replay Evidence `Latest run` from the local display values.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local latest-run id derivation and the
  absence of direct raw `latestRun?.runId` rendering.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to Replay Evidence presentation while global blockers are
visible. Rollback removes the local latest-run id binding, the M077 rails and
the M077 docs/state entries. M001 remains blocked on real WING read-only
evidence either way.
