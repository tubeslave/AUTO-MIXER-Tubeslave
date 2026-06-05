# Brief: deesser-proposal-freshness-gate-20260605

## Context

The project goal requires automatic de-essing, but current live authority does
not include de-esser, dynamic EQ, or dynamics writes. The de-esser proposal layer
uses vocal sibilance evidence, but it also needs to respect whether the current
de-esser/dynamics state cache is fresh enough to trust.

## Task

Add a read-only freshness gate to `deesser_vocal_sibilance`: when current
dynamics or dynamic-EQ state freshness is supplied and is not fresh, demote the
recommendation to `analysis_only` with an explicit blocker. Fresh state should
remain eligible for proposal-only output.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes.
- Do not add de-esser, EQ, compressor, or dynamic-EQ apply support.
- Keep de-esser output proposal-only and non-queueable.
- Do not query a console from the proposal module.

## Expected Evidence

- Fresh current-state freshness is recorded in de-esser proposal evidence.
- Stale current-state freshness blocks proposal-only output.
- `build_deesser_proposals` filters stale-state items.
- Proposal-only registry/contract metadata mentions dynamics/dynamic-EQ
  freshness.
