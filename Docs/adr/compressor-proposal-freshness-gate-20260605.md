# ADR: compressor-proposal-freshness-gate-20260605

## Context

Compressor proposals use audio dynamics metrics, existing gain reduction, and
headroom. They also need to account for the freshness of the current console
dynamics state. A stale cached compressor threshold or makeup value can make a
recommendation look safer than it is.

## Options considered

1. Keep compressor proposals based only on audio dynamics and headroom.
2. Gate compressor proposals when provided current-state freshness is not fresh.
3. Require physical compressor readback before any proposal-only output.

## Decision

Use option 2. The proposal module remains transport-free. Fresh current-state
cache is recorded as evidence; stale, unknown, or missing provided freshness
demotes the item to `analysis_only`.

## Why this won

It consumes the new read-only field-freshness DTO without expanding live write
authority. It prevents known stale compressor state from producing confident
operator-facing compressor moves.

## Rejected alternatives

Option 1 ignores available safety evidence. Option 3 is the future promotion
bar, but it would break legacy audio-only proposal generation before all callers
can pass readback freshness.

## Implementation plan

- Add optional `current_state_cache_freshness` to compressor analysis input.
- Normalize freshness into recommendation evidence.
- Add `compressor_current_state_cache_not_fresh` when provided freshness is not
  fresh.
- Update proposal contract metadata to list current dynamics freshness evidence.
- Add tests for fresh and stale current-state freshness.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/compressor_dynamics_proposal.py backend/processing_registry.py backend/processing_proposal_adapters.py tests/test_compressor_dynamics_proposal.py`
- Focused compressor and registry tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: callers that pass stale freshness produce fewer compressor proposals.
Rollback is removing the freshness field, blocker, tests, registry/contract
wording, and this ADR. No live write behavior is affected.
