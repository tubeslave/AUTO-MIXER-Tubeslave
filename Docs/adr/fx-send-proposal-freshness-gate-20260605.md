# ADR: fx-send-proposal-freshness-gate-20260605

## Context

FX send proposals use source role, arrangement, return headroom, feedback risk,
and intelligibility metrics. They also need to account for whether current FX
send/return state is fresh. A stale FX cache can make a recommendation look
safer than it is, especially when send levels, return routing, or FX slot state
have changed.

## Options considered

1. Keep FX send proposals based only on audio, headroom, and risk evidence.
2. Gate FX send proposals when provided current-state freshness is not fresh.
3. Require physical FX send/return readback before any proposal-only output.

## Decision

Use option 2. The proposal module remains transport-free. Fresh current-state
cache is recorded as evidence; stale, unknown, or missing provided freshness
demotes the item to `analysis_only`.

## Why this won

It consumes read-only freshness evidence without expanding live write authority.
It prevents known stale FX send/return state from producing confident FX moves.

## Rejected alternatives

Option 1 ignores available safety evidence. Option 3 is the future promotion
bar, but it would break legacy audio-only proposal generation before all callers
can pass readback freshness.

## Implementation plan

- Add optional `current_state_cache_freshness` to FX send analysis input.
- Normalize freshness into recommendation evidence.
- Add `fx_current_state_cache_not_fresh` when provided freshness is not fresh.
- Update proposal contract metadata to list current FX send/return freshness
  evidence.
- Add tests for fresh and stale current-state freshness.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/fx_send_proposal.py backend/processing_registry.py backend/processing_proposal_adapters.py tests/test_fx_send_proposal.py`
- Focused FX and registry tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: callers that pass stale freshness produce fewer FX proposals. Rollback is
removing the freshness field, blocker, tests, registry/contract wording, and
this ADR. No live write behavior is affected.
