# Brief: mix-variant-source-identity-gate-20260605

## Context

The project goal requires automatic genre/task mix variants. The current
`mix_variant_profile_proposal` layer is transport-free and proposal-only; it
uses genre, task, reference metrics, arrangement context, Ayaic level planes,
headroom, feedback risk, style-clone guards, and freshness evidence before
fan-out into bus/group balance, FX send, master-bus, and scene/preset proposal
layers.

Mix-variant profiles depend on trusted source and arrangement identity. If the
lead vocal, drums, bass, support, or FX-role context is held for operator
review, the variant profile should not fan out to downstream mix proposals.

The worktree is dirty from earlier proposal-only safety and freshness cycles.
This task must not revert unrelated local changes.

## Task

Wire `source_identity_downstream_gate` into `mix_variant_profile_proposal`.
Legacy callers without calibration should remain compatible, but explicit held,
blocked, unrecognized, or stale source identity replay context must block
mix-variant profile proposals.

## Safety constraints

- Do not add live OSC, WING, dLive, Mixing Station, profile apply, preset
  apply, scene recall, or follow-up queue import support.
- Do not create queueable live actions.
- Do not mutate classifier thresholds, long-term memory, or proposal queues.
- Preserve existing genre/task/reference/arrangement, style-clone, headroom,
  feedback, active-performance, and current-context freshness blockers.
- Keep mix variant output proposal-only until trusted source identity, fresh
  reference metrics, fresh arrangement context, operator review, and downstream
  fan-out proof exist.

## Acceptance criteria

- Existing mix-variant proposal path remains compatible without calibration.
- Held source identity blocks mix-variant profile output.
- Stale source identity replay context blocks mix-variant profile output.
- Proposal-only contract includes source identity downstream trust evidence.
- Genre/task capability row lists the shared source identity gate as a support
  path.
- Focused tests and the standard test command pass.
