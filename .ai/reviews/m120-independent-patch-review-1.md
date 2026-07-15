# Independent patch review 1 — M120

## Verdict

NOT ACCEPT.

## Findings

1. The initial invalidation predicate was not causally bound to
   `disconnect_wing`. It accepted the common readiness-invalidated envelope
   before or after the acknowledgement, so emergency-stop, watchdog or
   physical-console-change invalidation could be relabelled as expected
   teardown traffic. It also failed to reject `success=false` or a non-empty
   `error` on that envelope.
2. Teardown evidence used the session-wide readiness latch instead of the
   helper's disconnect-local observation. A latch set before teardown could
   therefore produce false `readiness_invalidation_observed=true` evidence.
3. Negative tests did not cover those two loopholes.

## Required revision

- Accept the disconnect invalidation only after a valid
  `connection_status/connected=false` acknowledgement.
- Match the backend's exact disconnect-specific message and reject failed or
  errored invalidations.
- Preserve disconnect-local observation separately from the pre-existing
  readiness latch.
- Add adversarial tests for emergency stop, watchdog, physical console
  change, failed/error invalidation, pre-ack invalidation and prior-latch
  evidence.

## Disposition

The implementation and ADR must be revised and independently reviewed again
before M120 can be accepted or committed.
