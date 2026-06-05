# ADR: manual-write-readback-freshness-proof-20260605

## Context

Supervised manual writes produce the evidence used by proof reports,
decision traces, linked proposal rollback logic, and autonomy readiness. A
cache-backed post-write readback should prove that the cache observed the write,
not merely that an old cached value happens to equal the target.

## Options considered

1. Keep cache readback proof based only on value equality.
2. Require cache readback timestamps to be newer than the write attempt.
3. Require physical readback for all verified proof.

## Decision

Use option 2. Physical readback remains verified proof. Cache readback verifies a
write only when the cache timestamp is fresh and updated after the write attempt.

## Why this won

It prevents overstated `verified_from_cache` proof without blocking the
supervised write after it has already occurred. Linked actions can still detect
partial proof and roll back through the existing path.

## Rejected alternatives

Option 1 is too weak for autonomy evidence. Option 3 is the future promotion
bar, but it would unnecessarily break current cache-backed dry-run and simulated
proof paths before all clients expose physical query parity.

## Implementation plan

- Add a minimum cache timestamp parameter to manual readback freshness.
- Mark cache freshness `not_after_write` when the cache timestamp predates the
  write attempt.
- Return `readback_cache_not_fresh` from the readback verifier for cache proof
  that is stale, unknown, missing, or not post-write.
- Include readback cache freshness in decision traces.
- Add regression tests for post-write stale cache proof demotion.

## Test plan

- Focused server manual gate tests.
- Proof runner and operator proposal tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `git diff --check`

## Risks and rollback

Risk: fake or cache-only clients that update values without timestamps now
produce partial proof. Mitigation: timestamped runtime clients and test fakes
should update `state_timestamps` whenever state changes. Rollback is removing
the minimum timestamp check, trace field, tests, and this ADR.
