# ADR: WING MVP Supervised Prestart Artifact-Root Marker Gate

## Context

Accepted read-only decision candidates now carry artifact-root validation through
the ledger, no-WING gate, decision-review report, and supervised proof plan. The
final supervised prestart gate still named only the prestart trace in the
operator sequence marker text.

## Options Considered

1. Keep artifact-root validity implicit in proof-plan summary fields.
2. Add artifact-root trace as an explicit supervised prestart gate marker.

## Decision

Use option 2. The final supervised prestart gate must explicitly require a valid
accepted-candidate artifact-root trace.

## Why This Won

The supervised WING boundary is operator-facing. The text, structured gate,
handoff packet, no-WING summary, and Project Director baseline should all say
the same thing before a physical proof can be planned.

## Rejected Alternatives

Implicit-only proof-plan fields were rejected because they require readers to
infer safety from lower-level summary fields.

## Implementation Plan

- Update the contract constant and structured expected gate.
- Update handoff/no-WING/Project Director tests.
- Update MVP/runbook/autoloop docs and the current blocker command.

## Test Plan

- Run targeted handoff/no-WING/rails/Project Director tests.
- Run focused WING/no-WING tests.
- Run full standard pytest.
- Run no-WING quick, memory lint, Project Director, and `git diff --check`.

## Risks And Rollback

Risk is exact-string fixture drift. Rollback is reverting the marker and
structured gate field, but that would restore the operator-facing mismatch.
