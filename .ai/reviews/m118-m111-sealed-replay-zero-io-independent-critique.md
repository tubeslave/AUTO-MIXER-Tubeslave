# Independent Critique of Codex Proposal: M118

Independent read-only Codex reviewer fallback; this is not a Kimi opinion.

## 1. Strongest part of the other proposal

It refuses to ignore `socket_inet_construct=1`, keeps every constructor and
transport method guarded, and turns the outer exact-zero expectation into an
assertion inside the isolated test.

## 2. Possible mistakes or blind spots

The initial replay-wide IPv6 override was too broad. It could suppress
IPv6-dependent project/dependency branches and make zero counters true only in
an artificial environment.

## 3. Where the solution is too complex, fragile, or costly

The minimal test-only change is appropriately small. Do not add production
dependency pins or relax durable-store ancestor permissions.

## 4. What should be kept from the other proposal

Keep the exact-zero verifier, fail-fast assertion, no-WING boundary, fresh
source challenge requirement and private runtime artifact root.

## 5. My revised recommendation

Use the scoped preload-and-restore variant: guards active first, capability
false only for urllib3 import, restoration in `finally`, and an explicit check
that project imports see the original capability.
