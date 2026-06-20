# ADR: Replay Evidence Channel Status Blocker Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed presentation hardening that prevents operator surfaces
from implying promotion readiness while the integration bundle blocks planning.

Replay Evidence shows a channel evidence row with the selected channel and a
status badge derived from channel `gainRideState` and `liveApplyAllowed`. The raw
status is useful, but it can still look like channel evidence is apply-ready
unless the visible global blocker is reflected there.

## Options Considered

1. Leave the Replay Evidence channel status unchanged.
2. Hide the channel evidence row while the integration bundle is blocked.
3. Keep the row visible and override only its local status badge.

## Decision

Use option 3. `ReplayEvidencePanel` derives local `channelStatusLabel` and
`channelStatusTone` from `integrationBundleApplyBlocked`. While blocked, the
channel status badge shows `bundle blocked` with a red tone. Otherwise it
preserves the raw channel `gainRideState` and live-apply tone.

This does not mutate channel evidence input, replay run input, decision trace
input, task data, backend status input, backend result input, safety monitor
input or readiness input.

## Why This Won

The operator keeps useful channel evidence context while the UI makes the global
promotion blocker explicit at the channel status point. It is local presentation
logic only and does not alter channel evidence records, backend contracts or
apply paths.

## Rejected Alternatives

Leaving the badge unchanged keeps a misleading status gap. Hiding the channel
evidence row removes useful offline evidence and is not required for safety
because the badge can fail closed locally.

## Implementation Plan

- Add local channel status display derivation in `ReplayEvidencePanel`.
- Render the channel evidence badge from the local display values.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local channel-status derivation.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to Replay Evidence presentation. Rollback removes the local
channel-status derivation, the M062 rails and the M062 docs/state entries. M001
remains blocked on real WING read-only evidence either way.
