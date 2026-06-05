# ADR: long-term-memory-source-identity-gate-20260605

## Context

Long-term memory learning proposals use verified decisions, offline replay,
telemetry sessions, evidence packets, engineer actions, accepted/rejected
outcomes, reference comparisons, genre and arrangement context, replay
correlation, outcome consistency, safety regression evidence, and current
context freshness. They do not mutate memory directly, but accepted learning
records can influence future proposal priority. Source identity replay
calibration now exists, so memory learning should not learn from explicitly
untrusted source identity context.

## Options considered

1. Keep memory learning using only outcome counts, replay correlation,
   consistency, safety regression, and freshness evidence.
2. Wire the shared source identity downstream gate into memory learning.
3. Require fresh source identity calibration for all memory-learning calls.
4. Duplicate trust-state parsing inside memory learning.

## Decision

Use option 2. Add optional source identity trust fields to memory-learning
analysis input and evaluate them through `source_identity_downstream_gate`.
Missing legacy calibration remains compatible, but explicit held, blocked,
unrecognized, or stale replay trust blocks memory-learning proposals.

## Why this won

This moves replay-calibrated identity into a long-term learning safety gate
without adding live authority or direct memory mutation. It keeps blocker names
and no-mutation evidence consistent with existing downstream consumers while
preserving the current proposal-only learning boundary.

## Rejected alternatives

Outcome-count-only learning was rejected because a wrong source identity can
poison future proposal priority. Requiring calibration everywhere was rejected
because existing callers still need a compatibility path. Duplicating trust
parsing was rejected because downstream modules should share trust semantics.

## Implementation plan

- Add optional source identity trust fields to
  `LongTermMemoryLearningInput`.
- Evaluate trust through the shared downstream gate.
- Add trust blockers and evidence to memory-learning recommendations.
- Mark source identity downstream trust as required proposal evidence.
- Add the shared trust gate as a long-term memory capability support path.
- Add focused tests for legacy-compatible, held, and stale trust states.
- Run focused and full validation.

## Test plan

```bash
PYTHONPATH=backend python3 -m py_compile \
  backend/long_term_memory_learning_proposal.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py
PYTHONPATH=backend python3 -m pytest \
  tests/test_long_term_memory_learning_proposal.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Risks and rollback

Risk: explicit held source identity may block useful learning records until
operators accept labels. Rollback is removing the optional trust fields, gate
call, contract evidence entry, capability support path, and focused tests. No
live transport, queue mutation, model mutation, or memory mutation path is
added.
