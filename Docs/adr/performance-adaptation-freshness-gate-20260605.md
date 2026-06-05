# ADR: performance-adaptation-freshness-gate-20260605

## Context

Performance adaptation proposals use baseline levels, current readback, drift,
persistence, cooldown, headroom, feedback/bleed risk, rollback, latency, and
operator intent. They also need to account for whether the current performance
context is fresh. A stale live trend can make a correction recommendation unsafe
even when the bounded delta and headroom checks pass.

## Options considered

1. Keep performance adaptation proposals based only on readback flags, trend,
   safety, cooldown, and rollback checks.
2. Gate performance adaptation proposals when provided current-state freshness
   is not fresh.
3. Require physical current readback and replay match before any proposal-only
   output.

## Decision

Use option 2. The proposal module remains transport-free. Fresh current-state
cache is recorded as evidence; stale, unknown, or missing provided freshness
demotes the item to `analysis_only`.

## Why this won

It consumes read-only freshness evidence without expanding live write authority.
It prevents known stale performance context from producing confident
concert-time adaptation recommendations.

## Rejected alternatives

Option 1 ignores available safety evidence. Option 3 is the future promotion
bar, but it would break legacy proposal generation before all callers can pass
readback freshness and replay proof.

## Implementation plan

- Add optional `current_state_cache_freshness` to performance adaptation
  analysis input.
- Normalize freshness into recommendation evidence.
- Add `performance_adaptation_current_state_cache_not_fresh` when provided
  freshness is not fresh.
- Update proposal contract metadata to list current performance context
  freshness evidence.
- Add tests for fresh and stale current-state freshness.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/performance_adaptation_proposal.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_performance_adaptation_proposal.py`
- Focused performance adaptation and registry tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: callers that pass stale freshness produce fewer performance adaptation
proposals. Rollback is removing the freshness field, blocker, tests,
registry/contract wording, and this ADR. No live write behavior is affected.
