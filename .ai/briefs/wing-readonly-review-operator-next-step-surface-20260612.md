# Brief: WING read-only review operator next step surface

## Context

The WING read-only preflight runner can fail closed before opening a backend
WebSocket when the expected offline plan is missing, invalid, or stale. Recent
runner summaries include `operator_next_step`, but downstream evidence/review
rows did not preserve that guidance for manual review.

## Task

Surface `operator_next_step` in read-only evidence records and manual review
reports without changing candidate acceptance, live authority, or WING contact
behavior.

## Acceptance

- Invalid runner summaries keep their operator next-step guidance in evidence.
- Review JSON and Markdown show the next step for blocked artifacts.
- Clean candidates remain read-only and do not gain live authority.
- Tests prove the review remains fail-closed.
