# Brief: mute-management-freshness-gate-20260605

## Context

The project goal requires console mute control, but current live authority does
not include mute/unmute writes. The mute management proposal layer uses source
role, current mute state, signal activity, idle duration, dropout risk, feedback
risk, operator intent, and manual mute authority. It also needs to respect
whether the current mute state cache is fresh enough to trust.

## Task

Add a read-only freshness gate to `mute_management_proposal`: when current mute
state freshness is supplied and is not fresh, demote the recommendation to
`analysis_only` with an explicit blocker. Fresh state should remain eligible for
proposal-only output.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes.
- Do not add mute or unmute apply support.
- Keep mute output proposal-only and non-queueable.
- Preserve manual mute authority for manually-muted and critical sources.
- Do not query a console from the proposal module.

## Expected Evidence

- Fresh current-state freshness is recorded in mute proposal evidence.
- Stale current-state freshness blocks proposal-only output.
- `build_mute_management_proposals` filters stale-state items.
- Proposal-only registry/contract metadata mentions current mute freshness.
