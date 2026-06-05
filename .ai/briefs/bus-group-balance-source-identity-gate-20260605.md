# Brief: bus-group-balance-source-identity-gate-20260605

## Context

The project goal requires automatic mix balance and full DCA, group, matrix,
and bus control. `bus_group_balance_proposal` already stays transport-free and
proposal-only, using Ayaic level planes, summed child loudness, current fader
readback, assignment confidence, true-peak headroom, main-bus headroom,
feedback risk, persistence, and current bus/group/DCA/matrix cache freshness.

Grouped balance still depends on trusted source identity: a drums group, BGV
group, FX return, DCA, or matrix balance move is only meaningful if the source
role and grouped assignment context are trustworthy. Source identity downstream
trust is already shared by multiple source-specific proposal layers.

The worktree is dirty from earlier proposal-only safety and freshness cycles.
This task must not revert unrelated local changes.

## Task

Wire `source_identity_downstream_gate` into `bus_group_balance_proposal`.
Legacy callers without calibration should remain compatible, but explicit held,
blocked, unrecognized, or stale source identity replay context must block
bus/group/DCA/matrix balance proposals.

## Safety constraints

- Do not add live OSC, WING, dLive, Mixing Station, DCA, group, matrix, bus, or
  fader writes.
- Do not create queueable live actions.
- Do not mutate classifier thresholds, long-term memory, or proposal queues.
- Preserve existing Ayaic, child-sum, assignment, true-peak, main-bus,
  feedback, fader-ceiling, and current-state freshness blockers.
- Keep bus/group/DCA/matrix output proposal-only until trusted source identity,
  fresh assignment/fader readback, rollback, and hardware proof exist.

## Acceptance criteria

- Existing bus/group balance proposal path remains compatible without
  calibration.
- Held source identity blocks bus/group balance proposal output.
- Stale source identity replay context blocks bus/group balance proposal output.
- Proposal-only contract includes source identity downstream trust evidence.
- Mix-balance and DCA/group/matrix capability rows list the shared source
  identity gate as a support path.
- Focused tests and the standard test command pass.
