# ADR: M120 HW01 teardown readiness invalidation

## 1. Context

An attended real-WING HW01 run on source
`45e2a2ae66853dc79a8db693f8ccfe15cbc67cc8` produced accepted route proof
`4937d06623084a2d98e7bc3f9b8f8a17`. It verified the exact WING Rack serial,
firmware family, USB/CoreAudio identity, two Main 47/48 route passes, fresh
signal, conservative true peak `-12.318 dBTP` and zero console mutation writes.

The runner nevertheless returned blocked. During teardown the backend first
broadcast `connection_status/connected=false`, then truthfully broadcast
`auto_soundcheck_status/readiness_invalidated`. The generic receiver's
duplicate-terminal grace window treats that second message as fatal during
active work, so the runner reported `wing_disconnect_failed:RuntimeError`
despite the physical disconnect and clean backend shutdown.

## 2. Options considered

1. Remove or suppress the backend readiness-invalidated broadcast.
2. Catch all teardown receiver errors after an accepted route artifact.
3. Add a generic receiver boolean that permits readiness invalidation.
4. Add a private disconnect-teardown receiver with an exact acceptance
   envelope.

## 3. Decision

Choose option 4.

`ProofOwnedBackendSession` receives disconnect teardown through a dedicated
private helper. The helper:

- uses the same strict WebSocket object parser and one bounded deadline;
- first requires exactly one valid `connection_status` with
  `connected is False` and no blocked/error result;
- only after that acknowledgement, accepts at most one exact
  `auto_soundcheck_status/readiness_invalidated` notification with both
  running flags false, the backend's disconnect-specific message, a readiness
  mapping, and no failed/error result;
- never treats it as acknowledgement;
- preserves the duplicate-grace interval after acknowledgement;
- rejects duplicate invalidations, duplicate disconnect terminals and every
  unrelated control/failure terminal;
- operates even if active work already latched readiness invalidation, because
  safety teardown must still finish;
- reports disconnect-local invalidation observation separately from any
  readiness latch that predates teardown.

The normal `_receive_terminal()` and active-session readiness latch remain
unchanged.

## 4. Why this won

The exception belongs to teardown semantics, not to generic terminal receipt.
An explicit helper prevents future Apply/Proof/Audit code from enabling a
boolean safety escape hatch while still preserving the backend's truthful
readiness invalidation for all clients.

## 5. Rejected alternatives

- Suppressing the backend broadcast would hide real readiness loss.
- Catching all receiver failures could accept a missing disconnect or protocol
  corruption.
- A generic opt-in is a smaller diff but expands the safety API and is easier
  to misuse in active control paths.
- Treating the route artifact alone as HW01 completion violates the mandatory
  clean-teardown contract.

## 6. Implementation plan

1. Add the exact invalidation predicate and teardown receiver.
2. Use it only for `disconnect_wing` in `_close_impl()`.
3. Include whether expected readiness invalidation was observed in teardown
   evidence.
4. Add a positive real-order test plus fail-closed tests for invalidation
   before acknowledgement, missing acknowledgement, duplicates, emergency
   stop, watchdog, physical console change, failed/error and malformed
   terminals.
5. Keep the existing active-session invalidation regression unchanged.
6. Validate, independently review, commit, rebind CP05 and repeat real HW01.

## 7. Test plan

- Focused new teardown tests.
- Complete `tests/test_wing_attended_soundcheck_runner.py`.
- Route-proof, CP05 and WING static rails.
- Full repository suite and no-WING release check.
- Real route-only repetition from clean committed source.

## 8. Risks and rollback

The main risk is accidentally tolerating a non-teardown safety invalidation.
The private call graph, acknowledgement-first ordering, disconnect-specific
message, local observation evidence and unchanged generic receiver contain
that risk. Rollback is a revert of the M120 source commit followed by a
marker-only CP05 rebind; no console state is changed by this software fix.
