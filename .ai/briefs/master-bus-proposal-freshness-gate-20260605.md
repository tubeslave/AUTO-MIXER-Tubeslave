# Brief: master-bus-proposal-freshness-gate-20260605

## Context

The project goal requires automatic master-bus optimization, but current live
authority does not include main-bus EQ, limiter, compressor, matrix, or master
processing writes. The master-bus proposal layer uses loudness, true-peak,
tonal, dynamics, stereo, and reference-profile evidence, but it also needs to
respect whether the current main-bus processing state cache is fresh enough to
trust.

## Task

Add a read-only freshness gate to `master_bus_optimization_proposal`: when
current main-bus processing state freshness is supplied and is not fresh, demote
the recommendation to `analysis_only` with an explicit blocker. Fresh state
should remain eligible for proposal-only output.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes.
- Do not add main-bus EQ, limiter, compressor, matrix, or master processing
  apply support.
- Keep master-bus output proposal-only and non-queueable.
- Do not query a console from the proposal module.

## Expected Evidence

- Fresh current-state freshness is recorded in master-bus proposal evidence.
- Stale current-state freshness blocks proposal-only output.
- `build_master_bus_proposals` filters stale-state items.
- Proposal-only registry/contract metadata mentions current main-bus processing
  freshness.
