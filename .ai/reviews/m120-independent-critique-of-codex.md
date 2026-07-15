# Independent critique of Codex M120 proposal

## 1. Strongest part of the Codex proposal

The separate `_receive_disconnect_terminal()` gives the safety exception an
explicit, auditable API boundary. It makes clear that readiness invalidation is
acceptable only as the inevitable side effect of a requested terminal
disconnect, never during Apply, Proof, Audit or other active control work.

## 2. Possible mistakes or blind spots

The helper must work even when the readiness latch was already set by the
active operation; teardown must still complete. It must not count invalidation
as the disconnect acknowledgement, and it must preserve one shared deadline,
duplicate detection and the generic fatal handling for every other control or
failure terminal.

## 3. Where the solution is too complex, fragile, or costly

A separate helper can duplicate demultiplexing behavior and drift from the
generic receiver. It should reuse `_recv_object()` and
`_reject_unexpected_terminal()` rather than implement a relaxed or simplified
WebSocket reader.

## 4. What should be kept

- Exact teardown-only API name.
- Mandatory `connection_status` with `connected is False`.
- Exact matching of the expected readiness invalidation.
- Existing active-session invalidation behavior and tests unchanged.

## 5. Revised recommendation

Choose the separate helper. It is safer than a boolean escape hatch in the
generic receiver, provided it retains the same deadline and duplicate rules,
rejects every unrelated terminal, and succeeds only after a real disconnect
acknowledgement.
