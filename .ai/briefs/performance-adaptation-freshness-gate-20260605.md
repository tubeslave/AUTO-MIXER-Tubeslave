# Brief: performance-adaptation-freshness-gate-20260605

## Context

The project goal requires automatic musician-level adaptation, concert-time
adaptation, and real-time corrections. The current implementation exposes these
as proposal-only performance adaptation recommendations. It uses baseline level,
current readback, drift, persistence, cooldown, headroom, feedback/bleed risk,
rollback, latency, and operator intent, but it should not trust stale live
trend/readback context.

## Task

Add a read-only freshness gate to `performance_adaptation_proposal`: when
current performance context freshness is supplied and is not fresh, demote the
recommendation to `analysis_only` with an explicit blocker. Fresh state should
remain eligible for proposal-only output.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes.
- Do not add fader/gain apply support from performance adaptation.
- Keep performance adaptation output proposal-only and non-queueable.
- Do not query a console from the proposal module.

## Expected Evidence

- Fresh current-state freshness is recorded in performance adaptation evidence.
- Stale current-state freshness blocks proposal-only output.
- `build_performance_adaptation_proposals` filters stale-state items.
- Proposal-only registry/contract metadata mentions current performance context
  freshness.
