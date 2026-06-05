# ADR: phase-aware-eq-guard-freshness-gate-20260605

## Context

Phase-aware EQ guard output uses multichannel audio, group membership, anchor
selection, low-frequency sum loss, coherence, bleed risk, and latency metadata
to block risky EQ boosts and surface phase candidates. A stale audio window or
phase measurement snapshot can create a false delay/polarity candidate or mask
real cancellation.

## Options considered

1. Keep phase-aware EQ guard output based only on current audio arrays and
   confidence thresholds.
2. Gate phase-aware EQ guard output when provided current-state freshness is not
   fresh.
3. Require physical delay/polarity readback before any dry-run guard output.

## Decision

Use option 2. The guard remains transport-free and dry-run. Fresh current-state
cache is recorded as metadata; stale, unknown, or missing provided freshness
returns `PHASE_UNRELIABLE` with no suggested actions.

## Why this won

It consumes read-only freshness evidence without expanding live write authority.
It prevents known stale phase evidence from producing candidates that could
drive unsafe downstream EQ decisions.

## Rejected alternatives

Option 1 ignores available safety evidence. Option 3 is the future promotion
bar, but it would break legacy guard analysis before all callers can pass
freshness and hardware proof.

## Implementation plan

- Add optional `current_state_cache_freshness` to phase-aware EQ guard config.
- Normalize freshness into result metadata.
- Return `PHASE_UNRELIABLE` with no suggested actions when provided freshness is
  not fresh.
- Update proposal contract metadata to list current phase/audio context
  freshness evidence.
- Add tests for fresh and stale current-state freshness.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/phase_aware_eq_guard.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_phase_aware_eq_guard.py`
- Focused phase-aware EQ guard and registry tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: callers that pass stale freshness will get no phase candidates until they
provide fresh phase context. Rollback is removing the freshness field, metadata,
blocker, tests, registry/contract wording, and this ADR. No live write behavior
is affected.
