# Brief: phase-aware-eq-guard-freshness-gate-20260605

## Context

The project goal requires automatic EQ, phase correction, frequency separation,
and cross-channel decisions. The current phase-aware EQ guard is transport-free
and dry-run: it analyzes multichannel phase relationships, blocks risky EQ
boosts, and emits phase/delay/polarity candidates for operator review. Those
candidates should not be produced from stale phase/audio context.

## Task

Add a read-only freshness gate to `phase_aware_eq_guard`: when current
phase/audio context freshness is supplied and is not fresh, return a
`PHASE_UNRELIABLE` result with no suggested actions and an explicit blocker in
metadata. Fresh context should remain eligible for proposal-only phase/EQ guard
output.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes.
- Do not add delay, polarity, EQ, gain, or fader apply support.
- Keep phase-aware EQ output dry-run/proposal-only.
- Do not query a console from the guard module.

## Expected Evidence

- Fresh current-state freshness is recorded in phase-aware EQ guard metadata.
- Stale current-state freshness blocks phase candidates and suggested actions.
- Proposal-only registry/contract metadata mentions current phase/audio context
  freshness.
- Full tests and diff hygiene remain clean.
