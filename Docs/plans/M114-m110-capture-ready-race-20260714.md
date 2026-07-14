# M114 — M110 Capture-Ready Race

Status: VALIDATED_FOR_COMMIT

## Purpose

Make the attended HW01 route proof deterministic when a valid WING CoreAudio
stream needs a bounded interval to deliver its first callback, without weakening
identity, ownership, signal, headroom or teardown rails.

## Sequence

1. Record the real fail-closed HW01 attempt and Council decision.
2. Add bounded same-WebSocket capture-readiness polling with strict identity
   pinning.
3. Add adversarial runner tests and obtain independent review.
4. Release the development lease and run focused and full validation.
5. Commit the non-marker tree, then rebind the two CP05 markers in a separate
   marker-only commit.
6. Repeat route-only HW01. Only a verified route/headroom result may proceed to
   the interactive reset and staged soundcheck flow.

## Exit criteria

- A delayed first callback can become ready within the fixed deadline.
- A dead callback, stopped stream, malformed status or identity drift blocks
  before route proof and performs clean teardown.
- Tests and independent review pass on committed source.
- CP05 verifies the final clean tracked tree.
- Real HW01 either issues independently verified route authority or records an
  exact remaining hardware blocker; no correction occurs before that result.

## Validation evidence

- Attended runner suite: `125 passed`.
- Combined runner, Main 47/48 route proof, M110 capability and M006 restore
  slice: `225 passed`.
- Cancellation teardown scenario: `5/5` repeated passes.
- Deadline/cadence regression scenarios: `10/10` repeated passes in independent
  review.
- Independent final review: `CLEAN`, P0/P1/P2 = 0.
- `git diff --check`: passed.
- Development writer lease: released; `ops/autopilot/STOP`: absent.
- WING rail verification is intentionally deferred until the non-marker commit
  and CP05 marker-only rebind because the current tracked tree is dirty by
  design during implementation.
