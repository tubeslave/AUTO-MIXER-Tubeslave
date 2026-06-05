# Brief: fx-send-source-identity-gate-20260605

## Context

The project goal requires automatic FX management and effect-send tuning, but
FX sends are source-specific. A wrong source identity label can add ambience or
delay to a dry anchor, vocal, or playback source in a way that hurts
intelligibility or feedback safety. Source identity downstream trust is already
shared by arrangement-role, frequency-separation, and de-esser proposal layers.
The FX send layer should consume the same gate.

The worktree is dirty from earlier proposal-only safety and freshness cycles.
This task must not revert unrelated local changes.

## Task

Wire `source_identity_downstream_gate` into `fx_send_proposal`. Legacy callers
without calibration should remain compatible, but explicit held, blocked,
unknown, or stale source identity replay context must block FX send proposals.

## Safety constraints

- Do not add live OSC, WING, dLive, or Mixing Station writes.
- Do not create queueable live actions.
- Do not mutate classifier thresholds, long-term memory, or proposal queues.
- Preserve existing FX current-state freshness, return-headroom, feedback, and
  intelligibility blockers.
- Keep FX send output proposal-only until send/return readback and rollback
  proof exist.

## Acceptance criteria

- Existing FX send proposal path remains compatible without calibration.
- Held source identity blocks FX send proposal output.
- Stale source identity replay context blocks FX send proposal output.
- Proposal-only contract includes source identity downstream trust evidence.
- Focused tests and the standard test command pass.
