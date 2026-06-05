# Brief: fx-send-proposal-freshness-gate-20260605

## Context

The project goal requires automatic effects management and effect-send tuning,
but current live authority does not include FX rack, send, return, or insert
writes. The FX send proposal layer uses source role, arrangement, return
headroom, feedback risk, and intelligibility evidence, but it also needs to
respect whether the current FX send/return state cache is fresh enough to trust.

## Task

Add a read-only freshness gate to `fx_send_proposal`: when current FX send/return
state freshness is supplied and is not fresh, demote the recommendation to
`analysis_only` with an explicit blocker. Fresh state should remain eligible for
proposal-only output.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes.
- Do not add FX rack, send, return, insert, or preset apply support.
- Keep FX output proposal-only and non-queueable.
- Do not query a console from the proposal module.

## Expected Evidence

- Fresh current-state freshness is recorded in FX proposal evidence.
- Stale current-state freshness blocks proposal-only output.
- `build_fx_send_proposals` filters stale-state items.
- Proposal-only registry/contract metadata mentions current FX send/return
  freshness.
