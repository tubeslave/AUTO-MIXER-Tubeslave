# Brief: performance-adaptation-source-identity-gate-20260605

## Context

The project goal requires musician-level adaptation, concert-time adaptation,
and real-time corrections. These decisions are source-specific: a level ride
for a lead source, backing source, high-bleed source, or critical anchor carries
different safety and mix consequences. Source identity downstream trust is
already shared by arrangement-role, frequency-separation, de-esser, FX send,
pan, and compressor proposal layers. The performance-adaptation layer should
consume the same gate before proposing source-specific rides.

The worktree is dirty from earlier proposal-only safety and freshness cycles.
This task must not revert unrelated local changes.

## Task

Wire `source_identity_downstream_gate` into
`performance_adaptation_proposal`. Legacy callers without calibration should
remain compatible, but explicit held, blocked, unknown, or stale source identity
replay context must block adaptation proposals.

## Safety constraints

- Do not add live OSC, WING, dLive, or Mixing Station writes.
- Do not create queueable live actions.
- Do not mutate classifier thresholds, long-term memory, or proposal queues.
- Preserve existing performance-context freshness, baseline/readback/rollback,
  cooldown, latency, headroom, feedback, bleed, and manual override blockers.
- Keep performance-adaptation output proposal-only until replay and hardware
  proof exist.

## Acceptance criteria

- Existing performance adaptation proposal path remains compatible without
  calibration.
- Held source identity blocks adaptation proposal output.
- Stale source identity replay context blocks adaptation proposal output.
- Proposal-only contract includes source identity downstream trust evidence.
- Focused tests and the standard test command pass.
