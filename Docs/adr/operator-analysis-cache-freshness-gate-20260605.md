# ADR: operator-analysis-cache-freshness-gate-20260605

## Context

Operator analysis creates fader/gain proposals from cached mixer state and live
audio measurements. Cached current fader/gain values can become stale, and stale
state must not be promoted as an applyable supervised action.

## Options considered

1. Let stale cache pass and rely only on apply-time pre-read.
2. Require fresh cache timestamps before operator-analysis proposals become
   applyable.
3. Query the console from operator analysis.

## Decision

Use option 2. Operator-analysis observations remain visible, but requested
fader/gain changes require fresh cache timestamp evidence before they are
applyable.

## Why this won

It adds a promotion gate before queue/apply while preserving the read-only
operator-analysis boundary and the existing supervised apply proof path.

## Rejected alternatives

Option 1 allows stale cache to become an applyable proposal. Option 3 violates
the read-only analysis boundary and risks hidden transport behavior.

## Implementation plan

- Add cache freshness helpers to `operator_analysis`.
- Attach freshness evidence to fresh requested changes.
- Strip requested changes and store blocker metadata for stale/unknown cache.
- Preserve `cache_freshness` in normalized proposal queue changes.
- Add focused regression tests.

## Test plan

- `PYTHONPATH=backend python -m pytest tests/test_operator_product_state.py -q`
- Broader operator/readiness/server tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `git diff --check`

## Risks and rollback

Risk: older runtime clients without timestamps may produce analysis-only
proposals until they observe fresh state. Mitigation: WING and dLive clients now
timestamp cache updates, and stale observations remain visible. Rollback is
removing the freshness gate, queue normalization field, tests, and docs.
