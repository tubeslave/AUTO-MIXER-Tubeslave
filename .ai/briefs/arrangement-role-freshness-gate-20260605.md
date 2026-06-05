# Brief: arrangement-role-freshness-gate-20260605

## Context

The project goal requires automatic arrangement-role detection. Arrangement
roles can influence balance, pan, FX, unmasking, mix variants, and live
adaptation. The current implementation exposes arrangement roles as
proposal-only recommendations using source type, inter-channel context,
reference roles, accepted history, persistence, ambiguity, and masking evidence.
Those recommendations should not trust stale arrangement or inter-channel
context.

## Task

Add a read-only freshness gate to `arrangement_role_proposal`: when current
arrangement-role context freshness is supplied and is not fresh, demote the
recommendation to `analysis_only` with an explicit blocker. Fresh context should
remain eligible for proposal-only output.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes.
- Do not add downstream balance, EQ, pan, FX, or role-learning apply support.
- Keep arrangement-role output proposal-only and non-queueable.
- Do not query a console from the proposal module.

## Expected Evidence

- Fresh current-state freshness is recorded in arrangement-role proposal
  evidence.
- Stale current-state freshness blocks proposal-only output.
- `build_arrangement_role_proposals` filters stale-context items.
- Proposal-only registry/contract metadata mentions current arrangement-role
  context freshness.
