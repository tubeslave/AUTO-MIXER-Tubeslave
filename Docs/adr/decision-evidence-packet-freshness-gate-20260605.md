# ADR: decision-evidence-packet-freshness-gate-20260605

## Context

Decision evidence packets tie audio analysis, psychoacoustic models,
statistical models, project history, reference comparison, previous outcomes,
inter-channel context, arrangement context, and genre context to operator-review
decisions. A packet can meet family-count and confidence gates while still being
unsafe if the underlying evidence snapshot is stale.

## Options considered

1. Keep decision evidence packets based only on evidence availability,
   confidence, safety, latency, and conflict checks.
2. Gate decision evidence packets when provided current-state freshness is not
   fresh.
3. Require replay learning proof before any proposal-only evidence packet
   output.

## Decision

Use option 2. The proposal module remains transport-free. Fresh current-state
cache is recorded as evidence; stale, unknown, or missing provided freshness
demotes the item to `analysis_only`.

## Why this won

It consumes read-only freshness evidence without expanding live write authority.
It prevents known stale evidence snapshots from producing confident review
packets that may influence downstream proposals or learning.

## Rejected alternatives

Option 1 ignores available safety evidence. Option 3 is the future promotion
bar, but it would break legacy evidence packet generation before all callers can
pass freshness and replay proof.

## Implementation plan

- Add optional `current_state_cache_freshness` to decision evidence analysis
  input.
- Normalize freshness into recommendation evidence.
- Add `decision_evidence_context_cache_not_fresh` when provided freshness is not
  fresh.
- Update proposal contract metadata to list current decision-evidence context
  freshness evidence.
- Add tests for fresh and stale current-state freshness.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/decision_evidence_packet_proposal.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_decision_evidence_packet_proposal.py`
- Focused decision evidence and registry tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: callers that pass stale freshness produce fewer decision evidence packets.
Rollback is removing the freshness field, blocker, tests, registry/contract
wording, and this ADR. No live write behavior is affected.
