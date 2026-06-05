# ADR: master-bus-proposal-freshness-gate-20260605

## Context

Master-bus proposals use loudness, true-peak, tonal, dynamics, stereo, and
reference-profile evidence. They also need to account for whether current
main-bus processing state is fresh. A stale main-bus cache can make a
recommendation look safer than it is, especially when limiter ceiling, broad
EQ, matrix routing, or bus compression have changed.

## Options considered

1. Keep master-bus proposals based only on audio, headroom, and reference
   evidence.
2. Gate master-bus proposals when provided current-state freshness is not fresh.
3. Require physical main-bus processing readback before any proposal-only output.

## Decision

Use option 2. The proposal module remains transport-free. Fresh current-state
cache is recorded as evidence; stale, unknown, or missing provided freshness
demotes the item to `analysis_only`.

## Why this won

It consumes read-only freshness evidence without expanding live write authority.
It prevents known stale main-bus state from producing confident master-bus moves.

## Rejected alternatives

Option 1 ignores available safety evidence. Option 3 is the future promotion
bar, but it would break legacy audio-only proposal generation before all callers
can pass readback freshness.

## Implementation plan

- Add optional `current_state_cache_freshness` to master-bus analysis input.
- Normalize freshness into recommendation evidence.
- Add `master_bus_current_state_cache_not_fresh` when provided freshness is not
  fresh.
- Update proposal contract metadata to list current main-bus processing
  freshness evidence.
- Add tests for fresh and stale current-state freshness.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/master_bus_optimization_proposal.py backend/processing_registry.py backend/processing_proposal_adapters.py tests/test_master_bus_optimization_proposal.py`
- Focused master-bus and registry tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: callers that pass stale freshness produce fewer master-bus proposals.
Rollback is removing the freshness field, blocker, tests, registry/contract
wording, and this ADR. No live write behavior is affected.
