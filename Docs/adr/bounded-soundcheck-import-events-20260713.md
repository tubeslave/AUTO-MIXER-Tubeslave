# Bounded automatic soundcheck import events

## Context

During the 2026-07-13 real WING proposal-only soundcheck, the automatic
`operator_proposals_imported` broadcast exceeded the receiving WebSocket frame
limit. Each import result repeated a full proposal including raw analyzer
evidence, so a normal client closed with code `1009` even though the console
remained in observation-only mode. The backend control-plane limit is 256 KiB.

## Options considered

1. Raise the WebSocket frame limit for every client.
2. Stop broadcasting import completion.
3. Broadcast compact result metadata and keep full proposals on the dedicated
   decision-queue surface.

## Decision

Use option 3 with a hard serialized size budget of 128 KiB. Automatic import
broadcasts retain an allowlisted, bounded set of status, counts, proposal IDs,
operator-facing fields, apply blockers, readiness, and trigger metadata. Raw
proposal evidence is omitted. If compact rows still exceed the budget, the
event falls back to counts only and points clients to `get_decision_queue` and
`get_auto_soundcheck_status` for full state.

## Why this won

The control-plane event remains useful and provably below half of the transport
ceiling without weakening any live safety rule or requiring clients to accept
exceptionally large frames. Full proposal records remain owned by the operator
proposal queue.

## Rejected alternatives

- A larger global frame limit hides unbounded payload growth and transfers the
  memory risk to every client.
- Dropping the event removes a useful lifecycle signal and makes import failures
  harder to diagnose.

## Implementation plan

- Compact automatic import results in
  `backend/handlers/soundcheck_handlers.py`.
- Measure the same escaped JSON bytes used by the WebSocket transport.
- Use a fixed-size defensive fallback if even the counts-only form violates the
  budget.
- Preserve the existing queue and all apply blockers.
- Track the still-unbounded full decision-queue snapshot as a separate P1.

## Test plan

- Run `PYTHONPATH=backend python3 -m pytest tests/test_soundcheck_handlers.py -q --tb=short`.
- Inject a proposal with 1 MiB of raw evidence and assert that the broadcast is
  under 10 KiB while retaining its ID and apply blockers.
- Inject 101 rows with long Cyrillic/emoji fields and assert the serialized
  escaped JSON is at most 128 KiB and uses the counts-only fallback.

## Risks and rollback

Clients that incorrectly depended on raw evidence inside the automatic
broadcast must query the decision queue. Rollback is the single helper call at
the broadcast site; it does not affect mixer transport, proposal storage, or
console authority.
