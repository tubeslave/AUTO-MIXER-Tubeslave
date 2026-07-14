# Independent critique of the Codex M111 proposal

## 1. Strongest part of the other proposal

The Codex proposal correctly refuses to mutate historical evidence and keeps
historical physical facts, current compatibility evidence and operator
identity as separate roots of trust.

## 2. Possible mistakes or blind spots

It leaves room for an in-process replay and for reusing the existing
`m005_current_release_accepted` result. Both blur the distinction between
native runtime provenance and a migration equivalence decision.

## 3. Where the solution is too complex, fragile, or costly

A generic migration parser for arbitrary old runs is unnecessarily broad.
M111 should bind the one owner-accepted artifact and exact hashes. A narrowly
sealed replay is easier to audit than a generic conversion framework.

## 4. What should be kept

Keep immutable byte identities, clean-current source binding, canonical apply
path replay, explicit owner challenge, all authority denials and mutation tests.

## 5. Revised recommendation

Use a route-specific migration schema and aggregate acceptance flag. Require a
proof-owned out-of-process child replay and serial-bound owner attestation.
Leave native M005 provenance false and M006 blocked until both are present.
