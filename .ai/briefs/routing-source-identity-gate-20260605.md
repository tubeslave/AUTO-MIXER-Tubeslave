# Brief: routing-source-identity-gate-20260605

## Context

The project goal includes automatic routing control, but route diffs can
rewrite broad console state. `routing_management_proposal` already stays
transport-free and proposal-only, requiring route readback, rollback evidence,
operator intent, activity guards, scope limits, and fresh cached route state.
It still accepts source-specific route proposals without consuming replay-
calibrated source identity trust.

The worktree is dirty from earlier proposal-only safety and freshness cycles.
This task must not revert unrelated local changes.

## Task

Wire `source_identity_downstream_gate` into `routing_management_proposal`.
Legacy callers without calibration should remain compatible, but explicit held,
blocked, unrecognized, or stale source identity replay context must block
routing proposals.

## Safety constraints

- Do not add live OSC, WING, dLive, Mixing Station, routing, scene, or patch
  writes.
- Do not create queueable live actions.
- Do not mutate classifier thresholds, long-term memory, or proposal queues.
- Preserve existing route readback, rollback, operator intent, activity, scope,
  proposed-source readback, and current-route freshness blockers.
- Keep routing output proposal-only until trusted source identity, fresh full
  route readback, rollback, scope limits, and hardware proof exist.

## Acceptance criteria

- Existing routing proposal path remains compatible without calibration.
- Held source identity blocks routing proposal output.
- Stale source identity replay context blocks routing proposal output.
- Proposal-only contract includes source identity downstream trust evidence.
- Capability matrix lists the shared source identity gate as a routing support
  path.
- Focused tests and the standard test command pass.
