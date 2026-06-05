# Brief: long-term-memory-source-identity-gate-20260605

## Context

The project goal requires long-term memory that improves decisions from
accumulated data, logs, tests, and previous concerts. The current
`long_term_memory_learning_proposal` layer is transport-free, proposal-only, and
does not mutate memory. It uses verified decisions, offline replay, telemetry,
evidence packets, engineer actions, operator outcomes, references, genre,
arrangement, safety regression evidence, and context freshness before exposing
a learning record for operator review.

Learning records can affect future proposal priority. They should not learn
from held, blocked, unrecognized, or stale source identity context.

The worktree is dirty from earlier proposal-only safety and freshness cycles.
This task must not revert unrelated local changes.

## Task

Wire `source_identity_downstream_gate` into
`long_term_memory_learning_proposal`. Legacy callers without calibration should
remain compatible, but explicit held, blocked, unrecognized, or stale source
identity replay context must block memory-learning proposals.

## Safety constraints

- Do not add live OSC, WING, dLive, or Mixing Station writes.
- Do not mutate project memory from the proposal module.
- Do not create queueable live actions or learning auto-apply support.
- Preserve existing evidence-count, replay-correlation, outcome-consistency,
  safety-regression, conflict, manual-override, active-performance, and
  freshness blockers.
- Keep memory learning output proposal-only until trusted source identity,
  fresh replay/outcome context, operator acceptance, and replay correlation are
  proven.

## Acceptance criteria

- Existing memory-learning proposal path remains compatible without
  calibration.
- Held source identity blocks memory-learning proposal output.
- Stale source identity replay context blocks memory-learning proposal output.
- Proposal-only contract includes source identity downstream trust evidence.
- Long-term memory capability row lists the shared source identity gate as a
  support path.
- Focused tests and the standard test command pass.
