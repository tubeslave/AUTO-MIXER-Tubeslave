# Proposal: manual-write-readback-freshness-proof-20260605

## Short thesis

Treat cache-backed readback as proof only when its timestamp was refreshed after
the supervised write attempt.

## What I understood

The current work is building a path toward autonomous fader/gain control. The
manual write gate already requires pre-read freshness, but post-write proof is
what determines whether an action is actually verified and whether linked
actions can continue safely. A stale cache value should not be enough for
`proof_status=verified`.

## Proposed solution

Record the UTC epoch immediately before calling the mixer write method. When
post-write readback falls back to mixer state cache, compute cache freshness with
that epoch as a minimum timestamp. If the cache timestamp is older than the write
attempt, mark the freshness status as `not_after_write` and return
`readback_cache_not_fresh` from the readback verifier. Physical readback remains
unchanged.

## Likely files to touch

- `backend/server.py`
- `backend/handlers/product_state_handlers.py`
- `tests/test_server.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Block the write after stale post-write readback. Rejected because the write
  already happened; the correct action is proof demotion and rollback handling.
- Require physical readback for all proof. Rejected for current client parity,
  but physical readback remains the strongest proof when available.
- Leave cache proof unchanged. Rejected because it can overstate readiness.

## Risks

- Tests and fake clients must update timestamps when they simulate state updates.
- Existing cache-only clients without timestamped post-write updates will report
  partial proof until they expose real readback evidence.

## Test plan

- Focused manual gate tests for fresh cache readback, stale post-write cache,
  and rollback partial proof.
- Broader server/operator/proof tests.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may argue for requiring physical readback before any verified proof. I
agree as a long-term promotion bar, but this step improves current cache-only
simulation and keeps physical proof as a separate higher gate.
