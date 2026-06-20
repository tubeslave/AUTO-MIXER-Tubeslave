# ADR: Replay Evidence Report Row Blocker Visibility Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed UI hardening that prevents replay evidence surfaces
from hiding missing proof while integration-bundle blockers are visible.

Replay Evidence already blocks report path text when the row renders. The row
still depends only on `latestRun`, so a blocked state without a replay run can
hide the report proof surface completely.

## Options Considered

1. Leave report-row visibility tied only to `latestRun`.
2. Always render the report row.
3. Render the report row while blocked or when a replay run exists.

## Decision

Use option 3. `TaskQueuePanels` derives `replayReportVisible` from
`integrationBundleApplyBlocked || Boolean(latestRun)`. While blocked, the report
row remains visible and uses the already-blocked `replayReportLabel` and
`replayReportClassName`. Otherwise it preserves the existing latest-run based
visibility.

This does not mutate replay report input, replay run input, decision trace
input, channel evidence input, task data, backend status input, backend result
input, safety monitor input or readiness input.

## Why This Won

The operator can see the missing report proof instead of losing the row during
a blocked state. The change is local presentation logic only and does not alter
replay artifacts, backend contracts, WING preflight behavior, or supervised
apply paths.

## Rejected Alternatives

Leaving visibility tied only to `latestRun` keeps a hidden-proof gap. Rendering
the row unconditionally adds non-blocked UI noise and is not required because
the blocker case is the safety-critical path.

## Implementation Plan

- Add local report-row visibility derivation in `TaskQueuePanels`.
- Render the Replay Evidence report row from the local visibility flag.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local report-row visibility derivation
  and the absence of direct `latestRun` conditional rendering.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to Replay Evidence presentation while global blockers are
visible. Rollback removes the local report-row visibility binding, the M080
rails and the M080 docs/state entries. M001 remains blocked on real WING
read-only evidence either way.
