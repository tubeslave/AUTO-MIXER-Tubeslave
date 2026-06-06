# Brief: operator-proposal-batch-preflight-20260606

## Context

The decision queue can now list apply-preflight candidates and a single pending
proposal can be dry-run checked through `preflight_operator_proposal_apply`.
Operators still need a way to check the current queue candidates in one
explicit request without mutating proposal state or writing to the console.

## Objective

Add a Product Layer batch preflight handler that dry-runs pending apply
candidates and returns ready/blocked counts plus per-proposal evidence.

## Safety Constraints

- Batch preflight must be explicit; `build_decision_queue` remains snapshot-only.
- Do not call `_apply_manual_console_write`.
- Do not reserve idempotency keys.
- Do not mark proposal status or move queue entries.
- Do not widen live write kinds beyond the existing supervised fader/gain gate.

## Expected Evidence

- Batch preflight checks pending `can_apply=true` proposals by default.
- Ready and blocked proposal results are both returned.
- Analysis-only proposals are skipped unless explicitly included.
- Queue state, idempotency storage, cumulative-delta history, and mixer writes
  remain unchanged by the batch preflight.
