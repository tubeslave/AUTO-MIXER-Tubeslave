# ADR: arrangement-role-freshness-gate-20260605

## Context

Arrangement-role proposals use source type, section, inter-channel context,
reference roles, accepted history, persistence, ambiguity, and masking evidence.
The resulting role labels can influence balance, pan, FX, unmasking, mix
variants, and live adaptation. A stale arrangement or inter-channel context can
make a role recommendation unsafe even when confidence scores are high.

## Options considered

1. Keep arrangement-role proposals based only on availability flags,
   confidence, persistence, ambiguity, and masking checks.
2. Gate arrangement-role proposals when provided current-state freshness is not
   fresh.
3. Require accepted-history replay proof before any proposal-only output.

## Decision

Use option 2. The proposal module remains transport-free. Fresh current-state
cache is recorded as evidence; stale, unknown, or missing provided freshness
demotes the item to `analysis_only`.

## Why this won

It consumes read-only freshness evidence without expanding live write authority.
It prevents known stale arrangement context from producing confident source-role
recommendations that can affect downstream mix decisions.

## Rejected alternatives

Option 1 ignores available safety evidence. Option 3 is the future promotion
bar, but it would break legacy proposal generation before all callers can pass
freshness and replay proof.

## Implementation plan

- Add optional `current_state_cache_freshness` to arrangement-role analysis
  input.
- Normalize freshness into recommendation evidence.
- Add `arrangement_role_context_cache_not_fresh` when provided freshness is not
  fresh.
- Update proposal contract metadata to list current arrangement-role context
  freshness evidence.
- Add tests for fresh and stale current-state freshness.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/arrangement_role_proposal.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_arrangement_role_proposal.py`
- Focused arrangement-role and registry tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: callers that pass stale freshness produce fewer arrangement-role
proposals. Rollback is removing the freshness field, blocker, tests,
registry/contract wording, and this ADR. No live write behavior is affected.
