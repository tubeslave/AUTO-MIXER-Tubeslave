# Brief: pan-proposal-freshness-gate-20260605

## Context

The project goal requires automatic panning, but current live authority does
not include pan writes. The spatial pan proposal layer uses source role,
stereo-pair, stage-position, and mono-compatibility evidence, but it also needs
to respect whether the current pan state cache is fresh enough to trust.

## Task

Add a read-only freshness gate to `pan_spatial_proposal`: when current pan state
freshness is supplied and is not fresh, demote the recommendation to
`analysis_only` with an explicit blocker. Fresh state should remain eligible for
proposal-only output.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes.
- Do not add pan apply support.
- Keep pan output proposal-only and non-queueable.
- Do not query a console from the proposal module.

## Expected Evidence

- Fresh current-state freshness is recorded in pan proposal evidence.
- Stale current-state freshness blocks proposal-only output.
- `build_pan_proposals` filters stale-state items.
- Proposal-only registry/contract metadata mentions current pan freshness.
