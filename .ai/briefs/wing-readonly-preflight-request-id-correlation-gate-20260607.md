# Brief: WING Read-Only Preflight Request-ID Correlation Gate - 2026-06-07

## Task

Tighten the WING read-only preflight proof artifact so every parameter readback
is correlated to the exact request that produced it.

## Context

The current MVP remains the supervised WING Rack proof. The first open blocker
is real WING read-only connection proof. The runner already verifies WING mode,
operator-confirmed channel scope, physical readback source, forbidden request
types, event-log ordering, and channel/value/address identity.

One proof gap remained: a physical-looking readback could match channel,
value type, and address while still lacking an explicit request correlation id.

## Safety Boundary

- No WING, OSC, or WebSocket transport is contacted by this planning slice.
- No live write path is widened.
- No readiness row or MVP blocker is closed.
- The next real step remains running the read-only preflight against the
  intended backend/WING session.

## Expected Output

- `query_mixer_parameter` requests include deterministic `request_id`.
- Backend read-only responses echo `request_id` for success and blocked reads.
- Runner blocks mismatched or missing response request ids.
- Offline evidence rejects artifacts whose event log loses or mismatches the
  request id.
- No-WING plan validation expects the request-id contract.
