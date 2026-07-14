# Codex Critique of Independent M114 Proposal

## 1. Strongest part of the other proposal

It correctly separates initial ownership from callback readiness and notices
that `capture_started` must be latched immediately after ownership, otherwise a
later identity failure may rely only on child-process shutdown instead of the
explicit preview stop protocol.

## 2. Possible mistakes or blind spots

Using the receiver's 25 ms duplicate-terminal grace as the polling cadence can
approach 40 requests per second. The backend allows 10 requests per second with
a finite burst, so a one-second not-ready device could be rate-limited before
the intended deadline.

## 3. Where the solution is too complex, fragile, or costly

Changing generic handler semantics or globally changing terminal duplicate
handling would expand this hardware-only fix. The runner can reserve the
existing grace inside its deadline without altering every receiver call.

## 4. What should be kept from the other proposal

Keep initial owner proof, early teardown latching, strict identity pinning,
callback age and sample progression, optional owner/preview drift checks, a
monotonic deadline and adversarial teardown tests.

## 5. My revised recommendation

Implement the independent contract in the runner, but add an explicit 100 ms
minimum interval between status requests. Reserve the 25 ms duplicate grace
inside the absolute one-second deadline and send no route request until the
fresh status passes.
