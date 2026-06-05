# Brief: pan-source-identity-gate-20260605

## Context

The project goal requires automatic panning and console pan control, but spatial
placement is source-specific. A wrong source identity label can center, spread,
or move the wrong source and damage lead focus, low-frequency stability, stereo
pairs, or mono compatibility. Source identity downstream trust is already shared
by arrangement-role, frequency-separation, de-esser, and FX send proposal
layers. The pan layer should consume the same gate.

The worktree is dirty from earlier proposal-only safety and freshness cycles.
This task must not revert unrelated local changes.

## Task

Wire `source_identity_downstream_gate` into `pan_spatial_proposal`. Legacy
callers without calibration should remain compatible, but explicit held,
blocked, unknown, or stale source identity replay context must block pan
proposals.

## Safety constraints

- Do not add live OSC, WING, dLive, or Mixing Station writes.
- Do not create queueable live actions.
- Do not mutate classifier thresholds, long-term memory, or proposal queues.
- Preserve existing pan current-state freshness, mono-compatibility, stereo-pair,
  stage-position, and delta blockers.
- Keep pan output proposal-only until pan readback and rollback proof exist.

## Acceptance criteria

- Existing pan proposal path remains compatible without calibration.
- Held source identity blocks pan proposal output.
- Stale source identity replay context blocks pan proposal output.
- Proposal-only contract includes source identity downstream trust evidence.
- Focused tests and the standard test command pass.
