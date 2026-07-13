# WebSocket ingress safety and emergency fast lane

## Context

The backend previously awaited every authenticated WebSocket frame directly in
the receive loop. A slow normal handler could delay a later emergency stop, and
authenticated clients had no bounded post-auth request rate or application
queue. Full request payloads were also written to INFO logs. The checkout was
already dirty when this change was implemented; unrelated changes were left in
place.

## Options considered

1. Keep serial dispatch and only add a message-rate check.
2. Run every handler concurrently.
3. Use one bounded FIFO worker for normal traffic and an exact emergency-only
   fast lane, with per-connection and global token buckets.

## Decision

Use option 3. Normal traffic is limited to 10 frames/second with burst 30 per
connection and 50 frames/second with burst 100 globally. It enters one FIFO of
size 16 per connection. Only exact `emergency_stop` and
`emergency_stop_agent` message types bypass the limiter and FIFO. Accepting the
first stop advances an ingress emergency generation; normal frames queued under
the previous generation are discarded. Duplicate stops are coalesced and do
not advance the system emergency generation.

When bearer authentication is enabled, a separate global 5 attempts/second,
burst-20 bucket is consumed before the authentication frame is parsed. Tokens
must be exact 43-character base64url strings before HMAC lookup, so invalid or
malformed-token floods cannot bypass the admission boundary.

Twenty rejected frames within five seconds close the connection with WebSocket
code 1013. The protocol layer uses a 256 KiB frame limit and queue size 16 on
both initial and port-cleanup server starts. The configured active-client limit
also includes clients waiting on the application authentication frame. Payload
contents are no longer logged at INFO.

## Why this won

A single normal worker preserves request order and avoids concurrent mutation
by ordinary handlers. The separate stop lane removes queue and limiter latency
from the safety action without widening priority to reset, rollback, apply, or
other write-capable messages. Generation stamping prevents already-buffered
work from resuming after a stop.

## Rejected alternatives

Rate limiting alone leaves emergency stop behind slow work. Fully concurrent
dispatch loses normal message ordering and increases write-state race risk.
Giving reset or rollback priority would widen the bypass surface beyond the two
fail-safe stop operations.

Disconnect stops queued normal frames but shields an already-running handler
until it finishes; transport loss must not inject cancellation between durable
prepare, physical readback, apply and rollback stages.

## Implementation plan

- Keep reusable token-bucket, rejection-window, message classification, and
  queued-frame types in `backend/ws_transport.py`.
- Enforce admission, normal FIFO processing, emergency scheduling, and protocol
  bounds in `backend/server.py`.
- Keep the live-console transport untouched and verify entirely with mocks.

## Test plan

- Unit-test burst/refill behavior, atomic local/global debit, rejection window,
  and exact emergency classification.
- Test FIFO ordering, emergency preemption, stale-frame discard, limiter bypass,
  close code 1013, active/authentication client admission, duplicate-stop
  generation stability, both `websockets.serve` paths, and payload-log removal.
- Run `tests/test_ws_transport.py`, `tests/test_server.py`, agent-handler tests,
  authentication tests, Python compilation, and `git diff --check`.

## Risks and rollback

An overloaded normal client may now lose rejected frames and eventually receive
1013; clients should back off and reconnect. Emergency frames remain available
under overload, but duplicates are intentionally coalesced after the latch is
active. Rollback is limited to the ingress additions in `backend/server.py`,
`backend/ws_transport.py`, and their tests; no mixer protocol or console state
is modified by this decision.
