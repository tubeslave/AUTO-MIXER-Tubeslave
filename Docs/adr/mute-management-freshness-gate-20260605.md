# ADR: mute-management-freshness-gate-20260605

## Context

Mute management proposals use source role, current mute state, signal activity,
idle duration, dropout risk, feedback risk, operator intent, and manual mute
authority. They also need to account for whether current mute state is fresh. A
stale mute cache can make a recommendation unsafe, especially when an operator
has manually muted a source or ownership changed after the cached snapshot.

## Options considered

1. Keep mute proposals based only on audio activity, role, and risk evidence.
2. Gate mute proposals when provided current-state freshness is not fresh.
3. Require physical mute readback before any proposal-only output.

## Decision

Use option 2. The proposal module remains transport-free. Fresh current-state
cache is recorded as evidence; stale, unknown, or missing provided freshness
demotes the item to `analysis_only`.

## Why this won

It consumes read-only freshness evidence without expanding live write authority.
It prevents known stale mute state from producing confident mute or unmute
recommendations.

## Rejected alternatives

Option 1 ignores available safety evidence. Option 3 is the future promotion
bar, but it would break legacy audio-only proposal generation before all callers
can pass readback freshness.

## Implementation plan

- Add optional `current_state_cache_freshness` to mute management analysis
  input.
- Normalize freshness into recommendation evidence.
- Add `mute_current_state_cache_not_fresh` when provided freshness is not fresh.
- Update proposal contract metadata to list current mute freshness evidence.
- Add tests for fresh and stale current-state freshness.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/mute_management_proposal.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_mute_management_proposal.py`
- Focused mute and registry tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: callers that pass stale freshness produce fewer mute proposals. Rollback is
removing the freshness field, blocker, tests, registry/contract wording, and
this ADR. No live write behavior is affected.
