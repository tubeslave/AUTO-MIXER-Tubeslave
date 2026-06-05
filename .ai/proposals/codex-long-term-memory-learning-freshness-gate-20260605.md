# Proposal: long-term-memory-learning-freshness-gate-20260605

## Short thesis

Use current memory-learning context freshness as a proposal-only gate before
showing long-term learning records for operator review.

## What I understood

The Automixer should improve from verified decisions, replay results,
telemetry, evidence packets, and operator outcomes. Learning from stale replay
or stale accepted/rejected outcomes would make future decisions less reliable.
The next useful step is to consume cached freshness evidence while keeping the
module transport-free, non-queueable, and non-mutating.

## Proposed solution

Extend `LongTermMemoryLearningInput` with optional
`current_state_cache_freshness`. Normalize that payload into recommendation
evidence. If it is provided and the status is not `fresh`, add
`long_term_memory_context_cache_not_fresh` and demote the recommendation to
`analysis_only`. Keep operator proposals without `requested_change` or
`requested_changes`, and keep `memory_mutation` false.

## Likely files to touch

- `backend/long_term_memory_learning_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_long_term_memory_learning_proposal.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Require freshness for every memory learning proposal immediately. Rejected to
  avoid breaking legacy analysis before all callers pass product snapshot
  freshness.
- Ignore freshness until memory writes exist. Rejected because stale learning
  evidence can already be identified and blocked at the review boundary.
- Mutate `.ai/memory/project.md` or runtime decision memory from this module.
  Rejected because learning records require operator acceptance and replay
  correlation before any memory influence.

## Risks

- Callers that provide stale freshness will produce fewer memory-learning
  proposals.
- Missing freshness remains a legacy gap until upstream integration is complete.

## Test plan

- Focused long-term memory learning proposal tests.
- Registry/capability contract tests.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may argue missing freshness should block all learning records. I agree for
promotion, but this cycle keeps legacy review compatible while blocking known
stale evidence.
