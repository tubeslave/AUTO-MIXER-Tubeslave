# Brief: long-term-memory-learning-freshness-gate-20260605

## Context

The project goal requires long-term memory that improves decisions from
accumulated data, logs, tests, and previous concerts. The current implementation
exposes memory learning as proposal-only records using verified decisions,
offline replay, telemetry, evidence packets, engineer actions, operator
outcomes, references, genre, arrangement, and safety regression evidence. A
learning record should not be reviewed from stale replay or outcome context.

## Task

Add a read-only freshness gate to `long_term_memory_learning_proposal`: when
current memory-learning context freshness is supplied and is not fresh, demote
the recommendation to `analysis_only` with an explicit blocker. Fresh context
should remain eligible for proposal-only output.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes.
- Do not mutate project memory from the proposal module.
- Do not add proposal queue import or learning auto-apply support.
- Keep memory learning output proposal-only and non-queueable.

## Expected Evidence

- Fresh current-state freshness is recorded in memory learning proposal
  evidence.
- Stale current-state freshness blocks proposal-only output.
- `build_long_term_memory_learning_proposals` filters stale-context items.
- Proposal-only registry/contract metadata mentions current memory-learning
  context freshness.
