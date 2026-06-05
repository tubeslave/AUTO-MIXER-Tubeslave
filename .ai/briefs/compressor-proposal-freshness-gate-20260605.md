# Brief: compressor-proposal-freshness-gate-20260605

## Context

The channel processing snapshot now exposes field-level cache freshness for
compressor state. Compressor recommendations should use that evidence before
claiming a proposal-only recommendation is safe to show as a concrete dynamics
move. A stale compressor threshold, ratio, or makeup cache can make the system
misread the current processing state.

## Task

Add a read-only freshness gate to `compressor_dynamics_proposal`: when current
compressor-state freshness is supplied and is not fresh, the analyzer must
demote the item to `analysis_only` with an explicit blocker. Fresh state should
remain eligible for proposal-only output. Missing freshness remains a tolerated
legacy path but is recorded in evidence.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes.
- Do not add compressor apply/queue support.
- Keep compressor output proposal-only and non-queueable.
- Do not query a console from the proposal module.

## Expected Evidence

- Fresh compressor-state freshness is recorded in proposal evidence.
- Stale compressor-state freshness blocks proposal-only output.
- `build_compressor_proposals` filters stale-state items.
- Proposal-only registry/contract metadata mentions current dynamics freshness.
