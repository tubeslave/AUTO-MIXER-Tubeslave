# ADR: Replay Evidence Report Path Blocker Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed presentation hardening that prevents operator surfaces
from implying promotion readiness while the integration bundle blocks planning.

Replay Evidence shows the latest replay report path after a run. The raw path is
useful for offline review, but it can still look like accepted replay evidence
unless the visible global blocker is reflected there.

## Options Considered

1. Leave the Replay Evidence report path unchanged.
2. Hide the report line while the integration bundle is blocked.
3. Keep the report line visible and override only its local text and styling.

## Decision

Use option 3. `ReplayEvidencePanel` derives local `replayReportLabel` and
`replayReportClassName` from `integrationBundleApplyBlocked`. While blocked, the
report line shows `Report: bundle blocked` with danger styling. Otherwise it
preserves the raw replay report path or `pending` fallback.

This does not mutate replay report input, channel evidence input, replay run
input, decision trace input, task data, backend status input, backend result
input, safety monitor input or readiness input.

## Why This Won

The operator keeps useful report context while the UI makes the global promotion
blocker explicit at the report path point. It is local presentation logic only
and does not alter replay records, report artifacts, backend contracts or apply
paths.

## Rejected Alternatives

Leaving the path unchanged keeps a misleading stale-evidence gap. Hiding the
report line removes useful offline context and is not required for safety
because the line can fail closed locally.

## Implementation Plan

- Add local report path display derivation in `ReplayEvidencePanel`.
- Render the report line from the local display values.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local report-path derivation.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to Replay Evidence presentation. Rollback removes the local
report-path derivation, the M063 rails and the M063 docs/state entries. M001
remains blocked on real WING read-only evidence either way.
