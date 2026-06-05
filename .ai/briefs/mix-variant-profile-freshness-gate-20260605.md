# Brief: mix-variant-profile-freshness-gate-20260605

## Context

The project goal requires automatic genre/task mix variants. The current
implementation exposes mix variants as proposal-only profiles that can fan out
into bus/group balance, FX send, master-bus, and scene/preset proposal layers.
Those fan-outs should not be based on stale reference metrics, arrangement
roles, current profile state, or context snapshots.

## Task

Add a read-only freshness gate to `mix_variant_profile_proposal`: when current
mix-variant context freshness is supplied and is not fresh, demote the
recommendation to `analysis_only` with an explicit blocker. Fresh context should
remain eligible for proposal-only output.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes.
- Do not add profile apply, preset apply, scene recall, or follow-up queue
  import support.
- Keep mix variant output proposal-only and non-queueable.
- Do not query a console from the proposal module.

## Expected Evidence

- Fresh current-state freshness is recorded in mix variant proposal evidence.
- Stale current-state freshness blocks proposal-only output.
- `build_mix_variant_profile_proposals` filters stale-context items.
- Proposal-only registry/contract metadata mentions current mix-variant context
  freshness.
