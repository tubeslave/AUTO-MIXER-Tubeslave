# ADR: recommendation-bridge-cache-freshness-gate-20260605

## Context

SafeGain and no-write soundcheck recommendations can feed the operator proposal
queue. They may include current trim/fader values, but stale cached console
state must not become an applyable supervised action.

## Options considered

1. Leave recommendation imports as-is.
2. Gate runtime recommendation imports on fresh cache timestamps.
3. Query the console from the recommendation bridge.

## Decision

Use option 2. Runtime mixer clients with `state` must provide fresh timestamped
fader/gain cache for applyable bridge proposals.

## Why this won

It keeps SafeGain and soundcheck recommendations visible while preventing stale
runtime cache from entering the applyable proposal path.

## Rejected alternatives

Option 1 weakens the new cache freshness safety model. Option 3 violates the
transport-free bridge boundary and can hide hardware reads inside adapters.

## Implementation plan

- Add bridge-local freshness helpers for fader/gain requested changes.
- Attach freshness evidence on fresh runtime cache.
- Convert stale/unknown/missing runtime cache recommendations to analysis-only.
- Pass `self.mixer_client` into soundcheck recommendation imports.
- Add focused regression tests.

## Test plan

- `PYTHONPATH=backend python -m pytest tests/test_operator_product_state.py tests/test_safe_gain_ensemble_bridge.py -q`
- Broad operator/server/readiness tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `git diff --check`

## Risks and rollback

Risk: runtime imports can queue analysis-only recommendations when cache
timestamps are missing. Mitigation: WING/dLive clients timestamp cache updates,
and supervised apply still performs pre-read/readback. Rollback is removing the
bridge gate, server argument, tests, and docs.
