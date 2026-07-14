# Independent Critique of Codex M114 Proposal

## 1. Strongest part of the other proposal

The runner-only, same-WebSocket and fixed-deadline approach is the smallest
change that addresses the observed race without widening live authority.

## 2. Possible mistakes or blind spots

The original proposal does not explicitly latch `capture_started` immediately
after the first ownership proof, require callback age and sample-counter growth,
or account for the receiver's duplicate-terminal grace inside the one-second
deadline. It must also reject optional `owned_by_requester=false` or
`preview=false` in a generic poll.

## 3. Where the solution is too complex, fragile, or costly

Do not add a second start, create a backend owner token for read-only polling or
change the shared audio handler. Existing owner state and the later route gate
already bind the same WebSocket and capture object.

## 4. What should be kept from the other proposal

Keep exact identity pinning, bounded polling, same-session operation, immediate
failure on drift and the full CP05 revalidation/rebind before another HW01.

## 5. My revised recommendation

Accept the Codex design with four additions: early teardown latching, strict
fresh callback evidence, positive sample progression, and a deadline that
includes send/receive/duplicate-grace time. Prove cancellation and timeout
teardown in tests.
