# M109-CP02 — byte-bounded decision queue wire contract

## Problem

`get_decision_queue` bounds rows but serializes complete proposal `raw` and
`result` objects. A single 1 MiB proposal therefore exceeds the 256 KiB control
plane even when `limit=1`. Mixing-agent pending actions and `agent_status` are
also unbounded. The browser loses close code 1009, reconnects, and can fall back
to mock-derived state.

The list response is also the current apply source. There is no immutable
proposal version/content digest binding between what the operator reviewed and
what `apply_operator_proposal` executes.

## Required outcome

Define and implement a versioned v2 list/detail/apply contract whose complete
serialized list payload is at most 128 KiB, keeps every safety blocker visible
or exactly accounted for, and fails closed on stale/missing review identity.

## Safety invariants

- Full server-side proposal remains the only apply source.
- Compact rows never become apply authority.
- Any apply-capable entrypoint requires proposal ID, immutable version and the
  reviewed content digest; legacy routing must not be a digest bypass.
- Global hazards and unresolved `can_apply=false` rows cannot be displaced by
  newer low-risk traffic.
- Cursor/detail errors perform no queue mutation, idempotency reservation,
  backend/WING contact or console send.
- Close 1009 is non-retryable and cannot substitute mock state for live state.

## Contract decisions required before code

1. Exact v2 request/response message names and fields.
2. Immutable review-content projection and digest canonicalization.
3. Proposal version lifecycle and retention/eviction behavior.
4. Snapshot revision and opaque cursor semantics.
5. Pinned ordering and the overflow response when pinned metadata alone exceeds
   the byte budget.
6. Complete-payload serialization measurement, including summaries and agent
   status.
7. Stable error taxonomy and legacy migration policy.
8. Frontend detail/review/apply flow and close-1009 state.

## Acceptance criteria

1. A proposal containing 1 MiB of irrelevant raw evidence yields a list
   response no larger than 128 KiB.
2. Long Cyrillic/emoji text also stays within the total byte budget.
3. Exact total/included/omitted/pinned counts remain correct.
4. Old global hazards and `can_apply=false` rows precede ordinary new rows.
5. Missing detail, stale cursor, missing digest, version mismatch and digest
   mismatch return distinct fail-closed errors with `no_console_send_proven`.
6. Both operator-queue and mixing-agent rows obey row and byte limits.
7. Actual `serialize_message(...).encode("utf-8")` is the size oracle.
8. Legacy apply cannot bypass review identity.
9. Close 1009 stops reconnect and exposes a real transport failure without
   mock fallback.
10. Existing normal list consumers have an explicit migration/compatibility
    path.

## Out of scope

- WING contact, reset, DSP writes or live-authority widening.
- Changing normal allowlist count from zero.
- Rendering or returning arbitrary unbounded raw evidence over WebSocket.
- Treating a compact list row as proof that a human reviewed full detail.

## Dirty-state note

Three byte-identical `reports/master_mix_eval_20260713T*.json` test side effects
remain untracked and are outside CP02.
