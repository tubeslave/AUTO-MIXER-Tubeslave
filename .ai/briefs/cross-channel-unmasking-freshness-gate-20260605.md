# Brief: cross-channel-unmasking-freshness-gate-20260605

## Context

The project goal requires automatic cross-channel unmasking and frequency
separation, but live EQ writes are not proven outside the supervised fader/gain
boundary. `cross_adaptive_eq` already computes mirror-EQ corrections from band
energy and channel priority, while the capability matrix still asks to route
unmasking output into linked proposals with rollback evidence.

The worktree is dirty from prior proposal-only safety and freshness cycles. This
brief depends on those local contracts and must not revert unrelated changes.

## Task

Add a read-only freshness gate and proposal-result wrapper to `cross_adaptive_eq`.
When current masking context freshness is supplied and is not fresh, no
unmasking proposal actions should be emitted. Fresh or legacy missing freshness
may produce linked cut/boost proposal records, but they must remain non-queueable
and non-live.

## Safety constraints

- Do not add live OSC, WING, dLive, or Mixing Station writes.
- Preserve the legacy `calculate_corrections` API for existing offline/live
  comparison paths.
- Require cached or physical EQ-band readback before any future promotion.
- Include rollback evidence in proposal-only action records.
- Keep direct EQ apply quarantined.

## Acceptance criteria

- Fresh current-state freshness is recorded in unmasking metadata.
- Stale current-state freshness blocks adjustment and proposal output.
- Linked cut/boost proposal actions include masking pair, bounded gain delta,
  sum-bus check requirement, EQ-band readback requirement, and rollback source.
- Registry, proposal-only contract, and capability matrix require fresh masking
  context before promotion.
- Focused tests and the standard test command pass.
