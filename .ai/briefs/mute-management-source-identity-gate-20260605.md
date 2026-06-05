# Brief: mute-management-source-identity-gate-20260605

## Context

The project goal requires mute control, but mute/unmute decisions are
high-risk source-specific console actions. A wrong source identity label can
treat a critical source as a safe idle source, or unmute the wrong source before
a cue. Source identity downstream trust is already shared by several
processing and adaptation proposal layers. Mute management should consume the
same gate before proposing source-specific mute state changes.

The worktree is dirty from earlier proposal-only safety and freshness cycles.
This task must not revert unrelated local changes.

## Task

Wire `source_identity_downstream_gate` into `mute_management_proposal`. Legacy
callers without calibration should remain compatible, but explicit held,
blocked, unknown, or stale source identity replay context must block mute-state
proposals.

## Safety constraints

- Do not add live OSC, WING, dLive, or Mixing Station writes.
- Do not create queueable live actions.
- Do not mutate classifier thresholds, long-term memory, or proposal queues.
- Preserve existing current-mute freshness, manual authority, dropout,
  feedback, upcoming cue, and critical-source blockers.
- Keep mute-management output proposal-only until intent/readback/rollback and
  hardware proof exist.

## Acceptance criteria

- Existing mute-management proposal path remains compatible without calibration.
- Held source identity blocks mute proposal output.
- Stale source identity replay context blocks mute proposal output.
- Proposal-only contract includes source identity downstream trust evidence.
- Focused tests and the standard test command pass.
