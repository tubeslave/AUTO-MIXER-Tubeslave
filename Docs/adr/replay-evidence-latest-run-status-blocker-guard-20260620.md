# ADR: Replay Evidence Latest Run Status Blocker Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed presentation hardening that prevents operator surfaces
from implying readiness while promotion planning is blocked.

Replay Evidence shows the latest offline replay run and its status. When the
offline integration bundle blocks promotion planning, a green or completed raw
run status can be misread as readiness unless the blocker is visible on that
badge too.

## Options Considered

1. Leave the latest-run badge unchanged.
2. Hide latest-run evidence while the integration bundle is blocked.
3. Keep latest-run evidence visible and override only the badge status locally.

## Decision

Use option 3. `ReplayEvidencePanel` derives local `latestRunStatusLabel` and
`latestRunStatusTone` from `integrationBundleApplyBlocked`. While blocked, the
Latest run badge shows `bundle blocked` with a red tone. Otherwise it preserves
the raw replay status and `toneForReplayStatus(...)`.

This does not mutate replay run input, decision trace input, task data, backend
status input, backend result input, safety monitor input or readiness input.

## Why This Won

The operator keeps useful offline replay context while the UI makes the global
promotion blocker explicit at the run-status point. It is a small local
presentation change and does not alter replay records, backend contracts or
apply paths.

## Rejected Alternatives

Leaving the badge unchanged keeps a misleading status gap. Hiding latest-run
evidence removes useful offline diagnostic context and is not needed for live
safety because the status can fail closed locally.

## Implementation Plan

- Add local latest-run status derivation in `ReplayEvidencePanel`.
- Render the latest-run badge from the local status values.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local latest-run status derivation.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to Replay Evidence presentation. Rollback removes the local
latest-run status derivation, the M060 rails and the M060 docs/state entries.
M001 remains blocked on real WING read-only evidence either way.
