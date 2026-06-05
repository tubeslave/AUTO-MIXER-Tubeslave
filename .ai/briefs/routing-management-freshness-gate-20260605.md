# Brief: routing-management-freshness-gate-20260605

## Context

The project goal requires console routing control, but current live authority
does not include routing writes, output patch changes, insert/send source
changes, or scene/snapshot recall. The routing proposal layer uses route diffs,
readback state, rollback routes, activity checks, scope limits, and operator
intent. It also needs to respect whether the current route state cache is fresh
enough to trust.

## Task

Add a read-only freshness gate to `routing_management_proposal`: when current
route state freshness is supplied and is not fresh, demote the recommendation to
`analysis_only` with an explicit blocker. Fresh state should remain eligible for
proposal-only output.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes.
- Do not add routing, patch, insert, send-source, scene, or snapshot apply
  support.
- Keep routing output proposal-only and non-queueable.
- Do not query a console from the proposal module.

## Expected Evidence

- Fresh current-state freshness is recorded in routing proposal evidence.
- Stale current-state freshness blocks proposal-only output.
- `build_routing_management_proposals` filters stale-state items.
- Proposal-only registry/contract metadata mentions current route freshness.
