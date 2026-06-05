# Brief: manual-write-readback-freshness-proof-20260605

## Context

The supervised manual gate now blocks stale cache pre-read before `fader` and
`gain` writes. Post-write proof still needs the same rigor. A cache readback can
match the target value while its timestamp predates the write attempt, which is
not strong enough evidence for autonomous promotion or linked-action safety.

## Task

Demote cache-backed post-write readback unless the cache timestamp was updated
after the write attempt. Physical readback remains valid proof. Cache readback
without a post-write timestamp should keep the result applied but mark proof as
partial.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes during implementation.
- No new write surfaces or autonomous promotion.
- Keep supervised live authority limited to `fader` and `gain`.
- Preserve rollback availability based on pre-read value even when proof is
  partial.

## Expected Evidence

- Cache readback updated after write remains `verified_from_cache`.
- Cache readback not updated after write becomes `readback_cache_not_fresh`.
- `proof_status` becomes `partial` when post-write cache proof is stale.
- Rollback readback uses the same proof-demotion rule.
