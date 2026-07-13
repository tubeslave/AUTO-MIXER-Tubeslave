# Bounded automatic soundcheck import events

## Context

During the 2026-07-13 real WING proposal-only soundcheck, the automatic
`operator_proposals_imported` broadcast exceeded the default 64 MiB WebSocket
frame limit. Each import result repeated a full proposal including raw analyzer
evidence, so a normal client closed with code `1009` even though the console
remained in observation-only mode.

## Options considered

1. Raise the WebSocket frame limit for every client.
2. Stop broadcasting import completion.
3. Broadcast compact result metadata and keep full proposals on the dedicated
   decision-queue surface.

## Decision

Use option 3. Automatic import broadcasts retain status, counts, proposal IDs,
operator-facing fields, apply blockers, readiness, and trigger metadata. Raw
proposal evidence is omitted from the broadcast. The event declares
`results_compacted`, `result_count`, `results_omitted_count`, and
`full_results_endpoint=get_decision_queue`.

## Why this won

The control-plane event remains useful and bounded without weakening any live
safety rule or requiring clients to accept exceptionally large frames. Full
proposal records remain owned by the operator proposal queue.

## Rejected alternatives

- A larger global frame limit hides unbounded payload growth and transfers the
  memory risk to every client.
- Dropping the event removes a useful lifecycle signal and makes import failures
  harder to diagnose.

## Implementation plan

- Compact automatic import results in
  `backend/handlers/soundcheck_handlers.py`.
- Preserve the existing queue and all apply blockers.
- Track the still-unbounded full decision-queue snapshot as a separate P1.

## Test plan

- Run `PYTHONPATH=backend python3 -m pytest tests/test_soundcheck_handlers.py -q --tb=short`.
- Inject a proposal with 1 MiB of raw evidence and assert that the broadcast is
  under 10 KiB while retaining its ID and apply blockers.

## Risks and rollback

Clients that incorrectly depended on raw evidence inside the automatic
broadcast must query the decision queue. Rollback is the single helper call at
the broadcast site; it does not affect mixer transport, proposal storage, or
console authority.
