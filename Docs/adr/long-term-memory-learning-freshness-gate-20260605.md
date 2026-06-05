# ADR: long-term-memory-learning-freshness-gate-20260605

## Context

Long-term memory learning proposals use verified decisions, offline replay,
telemetry sessions, evidence packets, engineer actions, accepted/rejected
outcomes, reference comparisons, genre and arrangement context, replay
correlation, outcome consistency, and safety regression evidence. A record can
meet count and confidence thresholds while still being unsafe if the replay or
outcome snapshot is stale.

## Options considered

1. Keep memory-learning proposals based only on evidence counts, replay
   correlation, outcome consistency, and safety regression checks.
2. Gate memory-learning proposals when provided current-state freshness is not
   fresh.
3. Require accepted memory writes and replay learning proof before any
   proposal-only learning record output.

## Decision

Use option 2. The proposal module remains transport-free and does not mutate
memory. Fresh current-state cache is recorded as evidence; stale, unknown, or
missing provided freshness demotes the item to `analysis_only`.

## Why this won

It consumes read-only freshness evidence without expanding live write or memory
write authority. It prevents known stale replay or outcome context from
producing learning records that could influence future proposal priority.

## Rejected alternatives

Option 1 ignores available safety evidence. Option 3 is the future promotion
bar, but it would break legacy learning-review generation before all callers can
pass freshness and replay proof.

## Implementation plan

- Add optional `current_state_cache_freshness` to long-term memory learning
  input.
- Normalize freshness into recommendation evidence.
- Add `long_term_memory_context_cache_not_fresh` when provided freshness is not
  fresh.
- Update proposal contract metadata to list current memory-learning context
  freshness evidence.
- Add tests for fresh and stale current-state freshness.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/long_term_memory_learning_proposal.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_long_term_memory_learning_proposal.py`
- Focused long-term memory learning and registry tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: callers that pass stale freshness produce fewer memory-learning proposals.
Rollback is removing the freshness field, blocker, tests, registry/contract
wording, and this ADR. No live write or memory mutation behavior is affected.
