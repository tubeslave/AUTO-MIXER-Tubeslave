# ADR: Replay Evidence Trace Detail Blocker Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed UI hardening that prevents replay evidence surfaces
from implying promotion readiness while integration-bundle blockers are
visible.

Replay Evidence already blocks its trace status badge, latest-run status/id,
policy row, channel row, report path and start action. The `Trace` detail text
still renders raw replay trace counts or a raw decision-trace action/delta,
which can make stale trace evidence look current while replay start is blocked.

## Options Considered

1. Leave the trace detail unchanged and rely on the blocked status badge.
2. Hide the `Trace` row while blocked.
3. Keep the row visible and override only the local trace-detail display.

## Decision

Use option 3. `TaskQueuePanels` derives `traceDetailLabel` and
`traceDetailClassName` from `integrationBundleApplyBlocked`. While blocked, the
row shows `No trace proof` with danger styling. Otherwise it preserves the raw
replay trace count, decision-trace action/delta or `No decision trace`.

This does not mutate decision trace input, replay run input, channel evidence
input, task data, backend status input, backend result input, safety monitor
input or readiness input.

## Why This Won

The operator keeps the Replay Evidence layout while the missing trace-proof
state is explicit. The change is local presentation logic only and does not
alter replay artifacts, backend contracts, WING preflight behavior, or
supervised apply paths.

## Rejected Alternatives

Leaving raw trace details visible keeps a stale-evidence gap. Hiding the row
removes useful context and is not required because the value can fail closed
locally.

## Implementation Plan

- Add local trace-detail display derivation in `TaskQueuePanels`.
- Render Replay Evidence `Trace` detail from the local display values.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local trace-detail derivation and the
  absence of direct raw trace-detail rendering.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to Replay Evidence presentation while global blockers are
visible. Rollback removes the local trace-detail binding, the M078 rails and
the M078 docs/state entries. M001 remains blocked on real WING read-only
evidence either way.
