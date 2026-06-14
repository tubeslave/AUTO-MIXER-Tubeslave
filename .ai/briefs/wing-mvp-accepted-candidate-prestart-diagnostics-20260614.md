# Brief: WING MVP Accepted-Candidate Prestart Diagnostics

## Task

Expose structured diagnostics for accepted WING read-only review source ids that
do not satisfy the accepted-candidate prestart contract.

## Why

The current gate blocks supervised fader/gain proof when accepted source ids do
not have a valid prestart trace, but a single boolean makes operator repair
ambiguous. The next safe step is to surface exact per-source reasons while
keeping the blocker active and granting no live authority.

## Scope

- Shared contract helper for candidate prestart contract violations.
- Review decision ledger/report fields.
- Supervised proof plan summary fields.
- no-WING summary rendering.
- Focused tests and operator-facing docs.

## Out of Scope

- No WING connection.
- No live writes.
- No readiness closure.
- No supervised proof execution.
