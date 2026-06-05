# Brief: source-identity-downstream-gate-20260605

## Context

Source identity replay calibration now produces operator-review trust states,
but downstream proposal families still need to consume those trust states before
they make source-specific decisions. Arrangement-role and frequency-separation
proposals are early downstream consumers because they can steer later EQ, level,
and mix-balance actions.

The worktree is dirty from earlier proposal-only safety and freshness cycles.
This task must not revert unrelated local changes.

## Task

Add a shared, transport-free `source_identity_downstream_gate` helper and wire it
into arrangement-role and frequency-separation proposals. Legacy callers without
calibration should remain compatible, but explicit `hold`, `blocked`, unknown,
or stale replay-calibration context must block downstream proposals.

## Safety constraints

- Do not add live OSC, WING, dLive, or Mixing Station writes.
- Do not create queueable live actions.
- Do not mutate classifier thresholds, memory, or proposal queues.
- Keep legacy missing calibration as compatible analysis input, but block
  explicit untrusted or stale calibration states.
- Preserve proposal-only payload shape.

## Acceptance criteria

- Shared helper returns trusted, legacy-unverified, held, blocked, and stale
  decisions with explicit evidence.
- Arrangement-role proposals record source identity trust evidence and block on
  held/blocked/stale states.
- Frequency-separation proposals prevent untrusted source identities from
  participating in lane proposals.
- Proposal-only contracts list source identity downstream trust evidence.
- Focused tests and the standard test command pass.
