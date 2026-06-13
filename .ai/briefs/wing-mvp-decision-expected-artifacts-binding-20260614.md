# Brief: WING decision expected-artifacts binding

## Context

The WING read-only candidate evidence now records whether real `summary.json`
and `events.jsonl` paths match the expected artifact globs from the offline
plan. The decision ledger still needs the same binding before an accepted
operator decision can become `accepted_decision_ready`.

## Task

Require accepted review source ids to carry valid expected-artifact binding in
the shared accepted-candidate prestart contract.

## Acceptance

- Accepted candidate source ids are valid only when expected summary/event
  artifact fields are present and both path-match flags are true.
- Decision reports render the expected artifact fields next to the prestart
  contract evidence.
- Missing expected-artifact binding keeps the candidate blocked and does not
  grant live authority.
- The first MVP blocker remains open until a real WING read-only artifact is
  captured and manually reviewed.
