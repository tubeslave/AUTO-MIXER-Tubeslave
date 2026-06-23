# Brief: WING MVP no-WING Review Decision Ledger Artifact Root Summary Surface

Date: 2026-06-14

## Objective

Make the no-WING release summary expose and validate the review-decision
ledger accepted-candidate artifact-root contract fields.

## Context

The review-decision ledger now computes accepted-candidate artifact-root
contract fields, but the no-WING review-decision summary path only surfaced
the accepted-candidate prestart contract fields. Operators could see the later
proof-plan artifact-root section, but the ledger section itself did not show
why a manual readiness candidate stayed blocked by artifact-root metadata.

## Safety Boundary

This is offline/read-only validation. It must not contact WING, mutate console
state, update readiness, or grant live authority.

## Acceptance

- no-WING review-decision snapshot detail carries accepted-candidate
  artifact-root contract fields.
- The snapshot validator fails closed when accepted artifact-root contract is
  invalid but the ledger lacks the artifact-root blocker or still reports a
  manual readiness candidate.
- `summary.json` and operator-readable `summary.md` expose the ledger
  artifact-root contract fields.
