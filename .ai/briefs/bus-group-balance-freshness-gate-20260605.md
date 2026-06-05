# Brief: bus-group-balance-freshness-gate-20260605

## Context

The project goal requires automatic mix balance and full DCA, group, matrix, and
bus control, but current live authority does not include bus/group/DCA/matrix
writes. The bus/group balance proposal layer uses Ayaic level planes, summed
child loudness, current fader readback, assignment confidence, true-peak
headroom, main-bus headroom, feedback risk, and persistence. It also needs to
respect whether the current bus/group/DCA/matrix assignment and fader state
cache is fresh enough to trust.

## Task

Add a read-only freshness gate to `bus_group_balance_proposal`: when current
bus/group/DCA/matrix state freshness is supplied and is not fresh, demote the
recommendation to `analysis_only` with an explicit blocker. Fresh state should
remain eligible for proposal-only output.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes.
- Do not add bus, group, DCA, matrix, or assignment apply support.
- Keep bus/group balance output proposal-only and non-queueable.
- Do not query a console from the proposal module.

## Expected Evidence

- Fresh current-state freshness is recorded in bus/group balance proposal
  evidence.
- Stale current-state freshness blocks proposal-only output.
- `build_bus_group_balance_proposals` filters stale-state items.
- Proposal-only registry/contract metadata mentions current bus/group/DCA/matrix
  freshness.
