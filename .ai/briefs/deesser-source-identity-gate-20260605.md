# Brief: deesser-source-identity-gate-20260605

## Context

The project goal requires automatic de-essing, but de-essing is strongly
source-specific. A wrong source identity label can turn cymbal/overhead
sibilance into an unsafe vocal processing recommendation. Source identity
downstream trust is already wired into arrangement-role and frequency-separation
proposals; the vocal de-esser layer should consume the same gate.

The worktree is dirty from earlier proposal-only safety and freshness cycles.
This task must not revert unrelated local changes.

## Task

Wire `source_identity_downstream_gate` into `deesser_vocal_sibilance`. Legacy
callers without calibration should remain compatible, but explicit held,
blocked, unknown, or stale source identity replay context must block de-esser
proposals.

## Safety constraints

- Do not add live OSC, WING, dLive, or Mixing Station writes.
- Do not create queueable live actions.
- Do not mutate classifier thresholds, long-term memory, or proposal queues.
- Preserve existing de-esser freshness, true-peak, and non-vocal blockers.
- Keep de-esser output proposal-only until dynamics/dynamic-EQ readback and
  rollback proof exist.

## Acceptance criteria

- Existing vocal de-esser proposal path remains compatible without calibration.
- Held source identity blocks de-esser proposal output.
- Stale source identity replay context blocks de-esser proposal output.
- Proposal-only contract includes source identity downstream trust evidence.
- Focused tests and the standard test command pass.
