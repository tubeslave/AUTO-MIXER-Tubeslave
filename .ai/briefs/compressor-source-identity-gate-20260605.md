# Brief: compressor-source-identity-gate-20260605

## Context

The project goal requires automatic compression and console compressor control,
but dynamics settings are source-specific. A wrong source identity label can
select the wrong ratio, attack, release, target gain reduction, or makeup
behavior before any live compressor write exists. Source identity downstream
trust is already shared by arrangement-role, frequency-separation, de-esser, FX
send, and pan proposal layers. The compressor layer should consume the same
gate.

The worktree is dirty from earlier proposal-only safety and freshness cycles.
This task must not revert unrelated local changes.

## Task

Wire `source_identity_downstream_gate` into
`compressor_dynamics_proposal`. Legacy callers without calibration should
remain compatible, but explicit held, blocked, unknown, or stale source identity
replay context must block compressor proposals.

## Safety constraints

- Do not add live OSC, WING, dLive, or Mixing Station writes.
- Do not create queueable live actions.
- Do not mutate classifier thresholds, long-term memory, or proposal queues.
- Preserve existing compressor current-state freshness, true-peak headroom,
  existing gain-reduction, signal-floor, and bounded makeup blockers.
- Keep compressor output proposal-only until dynamics readback and rollback
  proof exist.

## Acceptance criteria

- Existing compressor proposal path remains compatible without calibration.
- Held source identity blocks compressor proposal output.
- Stale source identity replay context blocks compressor proposal output.
- Proposal-only contract includes source identity downstream trust evidence.
- Focused tests and the standard test command pass.
