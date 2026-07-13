# Codex critique of the independent readiness proposal

## 1. Strongest part of the other proposal

It correctly separates physical proof from standing authority and refuses to
let a mutable runtime log or report prose act as the promotion gate.

## 2. Possible mistakes or blind spots

Demoting every family without preserving the proof lifecycle could erase useful
operator evidence in the UI. The code should keep `hardware_proven` visible and
reviewable while normal execution remains blocked.

## 3. Where the solution is too complex, fragile, or costly

A new evidence exporter is useful but not required to close the immediate live
bypass. It should follow the runtime gate and test isolation fix, not delay them.

## 4. What should be kept from the other proposal

- Dry-run proof tool with typed live confirmation.
- Temporary test audit roots.
- Run-scoped immutable evidence instead of the 110 MB global log.
- Truth-surface reconciliation after executable behavior is safe.

## 5. My revised recommendation

Demote all currently unpromoted families to `hardware_proven` with no normal
live kinds, retain their proof metadata for review, close partial-write failure
paths, then add a validated promotion input as a separate checkpoint.
