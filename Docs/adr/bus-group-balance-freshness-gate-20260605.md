# ADR: bus-group-balance-freshness-gate-20260605

## Context

Bus/group balance proposals use Ayaic level planes, summed child loudness,
current fader readback, assignment confidence, true-peak headroom, main-bus
headroom, feedback risk, and persistence. They also need to account for whether
current bus/group/DCA/matrix state is fresh. A stale grouped-state cache can
make a recommendation look safer than it is, especially when assignments,
matrix sends, or fader positions have changed.

## Options considered

1. Keep bus/group balance proposals based only on loudness, headroom, and
   assignment confidence.
2. Gate bus/group balance proposals when provided current-state freshness is not
   fresh.
3. Require physical bus/group/DCA/matrix readback before any proposal-only
   output.

## Decision

Use option 2. The proposal module remains transport-free. Fresh current-state
cache is recorded as evidence; stale, unknown, or missing provided freshness
demotes the item to `analysis_only`.

## Why this won

It consumes read-only freshness evidence without expanding live write authority.
It prevents known stale bus/group/DCA/matrix state from producing confident
grouped balance moves.

## Rejected alternatives

Option 1 ignores available safety evidence. Option 3 is the future promotion
bar, but it would break legacy audio-only proposal generation before all callers
can pass readback freshness.

## Implementation plan

- Add optional `current_state_cache_freshness` to bus/group balance analysis
  input.
- Normalize freshness into recommendation evidence.
- Add `bus_group_current_state_cache_not_fresh` when provided freshness is not
  fresh.
- Update proposal contract metadata to list current bus/group/DCA/matrix
  freshness evidence.
- Add tests for fresh and stale current-state freshness.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/bus_group_balance_proposal.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_bus_group_balance_proposal.py`
- Focused bus/group balance and registry tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: callers that pass stale freshness produce fewer bus/group balance
proposals. Rollback is removing the freshness field, blocker, tests,
registry/contract wording, and this ADR. No live write behavior is affected.
