# ADR: mix-variant-profile-freshness-gate-20260605

## Context

Mix variant proposals use genre, task, reference metrics, arrangement context,
Ayaic level planes, headroom, feedback risk, and style-clone guards. They also
fan out into bus/group balance, FX send, master-bus, and scene/preset proposal
layers. A stale reference or arrangement context can make the entire fan-out
unsafe even when each downstream layer remains proposal-only.

## Options considered

1. Keep mix variant proposals based only on reference/arrangement availability,
   confidence, headroom, and style-clone guards.
2. Gate mix variant proposals when provided current-state freshness is not
   fresh.
3. Require full physical readback and downstream fan-out proof before any
   proposal-only output.

## Decision

Use option 2. The proposal module remains transport-free. Fresh current-state
cache is recorded as evidence; stale, unknown, or missing provided freshness
demotes the item to `analysis_only`.

## Why this won

It consumes read-only freshness evidence without expanding live write authority.
It prevents known stale reference or arrangement context from producing profile
recommendations that can fan out into multiple downstream proposal layers.

## Rejected alternatives

Option 1 ignores available safety evidence. Option 3 is the future promotion
bar, but it would break legacy proposal generation before all callers can pass
readback freshness and downstream fan-out proof.

## Implementation plan

- Add optional `current_state_cache_freshness` to mix variant profile analysis
  input.
- Normalize freshness into recommendation evidence.
- Add `mix_variant_context_cache_not_fresh` when provided freshness is not
  fresh.
- Update proposal contract metadata to list current mix-variant context
  freshness evidence.
- Add tests for fresh and stale current-state freshness.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/mix_variant_profile_proposal.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_mix_variant_profile_proposal.py`
- Focused mix variant and registry tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: callers that pass stale freshness produce fewer mix variant proposals.
Rollback is removing the freshness field, blocker, tests, registry/contract
wording, and this ADR. No live write behavior is affected.
