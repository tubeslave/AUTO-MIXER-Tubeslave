# ADR: routing-management-freshness-gate-20260605

## Context

Routing proposals use route diffs, readback state, rollback routes, activity
checks, scope limits, and operator intent. They also need to account for whether
current route state is fresh. A stale route cache can make a routing diff unsafe,
especially when source assignments, rollback routes, or patch snapshots have
changed.

## Options considered

1. Keep routing proposals based only on route diffs, readback flags, rollback,
   and scope checks.
2. Gate routing proposals when provided current-state freshness is not fresh.
3. Require physical full-route readback before any proposal-only output.

## Decision

Use option 2. The proposal module remains transport-free. Fresh current-state
cache is recorded as evidence; stale, unknown, or missing provided freshness
demotes the item to `analysis_only`.

## Why this won

It consumes read-only freshness evidence without expanding live write authority.
It prevents known stale route state from producing confident routing
recommendations.

## Rejected alternatives

Option 1 ignores available safety evidence. Option 3 is the future promotion
bar, but it would break legacy proposal generation before all callers can pass
readback freshness.

## Implementation plan

- Add optional `current_state_cache_freshness` to routing management analysis
  input.
- Normalize freshness into recommendation evidence.
- Add `routing_current_state_cache_not_fresh` when provided freshness is not
  fresh.
- Update proposal contract metadata to list current route freshness evidence.
- Add tests for fresh and stale current-state freshness.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/routing_management_proposal.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_routing_management_proposal.py`
- Focused routing and registry tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: callers that pass stale freshness produce fewer routing proposals. Rollback
is removing the freshness field, blocker, tests, registry/contract wording, and
this ADR. No live write behavior is affected.
