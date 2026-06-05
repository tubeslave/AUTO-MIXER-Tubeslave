# ADR: pan-proposal-freshness-gate-20260605

## Context

Pan proposals use source role, stereo-pair, stage-position, and
mono-compatibility metrics. They also need to account for whether current pan
state is fresh. A stale pan cache can make a recommendation look safer than it
is, especially when stereo links or manual pan moves have changed.

## Options considered

1. Keep pan proposals based only on spatial and mono-compatibility evidence.
2. Gate pan proposals when provided current-state freshness is not fresh.
3. Require physical pan readback before any proposal-only output.

## Decision

Use option 2. The proposal module remains transport-free. Fresh current-state
cache is recorded as evidence; stale, unknown, or missing provided freshness
demotes the item to `analysis_only`.

## Why this won

It consumes read-only freshness evidence without expanding live write authority.
It prevents known stale pan state from producing confident pan moves.

## Rejected alternatives

Option 1 ignores available safety evidence. Option 3 is the future promotion
bar, but it would break legacy audio-only proposal generation before all callers
can pass readback freshness.

## Implementation plan

- Add optional `current_state_cache_freshness` to pan analysis input.
- Normalize freshness into recommendation evidence.
- Add `pan_current_state_cache_not_fresh` when provided freshness is not fresh.
- Update proposal contract metadata to list current pan freshness evidence.
- Add tests for fresh and stale current-state freshness.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/pan_spatial_proposal.py backend/processing_registry.py backend/processing_proposal_adapters.py tests/test_pan_spatial_proposal.py`
- Focused pan and registry tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: callers that pass stale freshness produce fewer pan proposals. Rollback is
removing the freshness field, blocker, tests, registry/contract wording, and
this ADR. No live write behavior is affected.
