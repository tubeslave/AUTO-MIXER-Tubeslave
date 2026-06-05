# Brief: frequency-separation-proposal-20260605

## Context

The project goal requires automatic frequency separation of instruments. The
capability matrix already tracks the requirement, but it did not have a
dedicated analyzer for source/stem contribution evidence. Existing
`cross_adaptive_eq` work handles masking pairs; this task adds a complementary
proposal-only frequency-lane layer.

The worktree is dirty from earlier proposal-only safety and freshness cycles.
This task must not revert unrelated local changes.

## Task

Add a transport-free `frequency_separation_proposal` analyzer. It should detect
source/stem band-lane conflicts from band energy, source role, priority,
confidence, and contribution evidence. Fresh context may emit operator-facing
frequency-lane proposals. Stale provided context must block proposal output.

## Safety constraints

- Do not add live OSC, WING, dLive, or Mixing Station writes.
- Do not create queueable live actions.
- Require fresh source/stem spectrum context before proposal promotion.
- Require EQ-band/stem-spectrum readback and rollback evidence before any future
  derived EQ or fader move.
- Keep direct EQ, fader, and routing changes outside this module.

## Acceptance criteria

- Fresh current-state freshness is recorded in frequency-separation metadata.
- Stale current-state freshness blocks frequency-lane recommendations.
- Proposal records include source/stem contribution, lane-conflict pair,
  bounded subtractive delta, sum-bus check requirement, EQ-band readback
  requirement, and rollback source.
- Registry, proposal-only contract, and capability matrix reference the new
  analyzer.
- Focused tests and the standard test command pass.
