# Brief: operator-proposal-apply-preflight-20260606

## Context

Imported and operator-authored proposals can now reach the native decision
queue. The next safety gap is explainability before apply: operators need to
know whether a pending proposal would pass the supervised apply gate without
actually reserving an idempotency key, mutating the queue, or writing the
console.

## Objective

Add a Product Layer dry-run preflight handler for `apply_operator_proposal`
that reuses the existing supervised validation boundary and returns ready or
blocked evidence without live writes.

## Safety Constraints

- Do not call `_apply_manual_console_write`.
- Do not mark proposal status or move entries between pending/history.
- Do not reserve idempotency keys.
- Do not widen live write kinds beyond fader/gain.
- Preserve approval, idempotency, operator mode, supported kind, connection,
  pre-read, delta, upward-move, and cumulative-delta gates.

## Expected Evidence

- A supervised fader/gain proposal with fresh pre-read returns ready evidence.
- Missing/stale pre-read and unsupported action kinds return blocked evidence.
- Preflight calls do not mutate queue state, idempotency storage, or live mixer
  state.
