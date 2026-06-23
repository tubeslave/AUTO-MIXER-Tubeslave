# Brief: WING MVP review decision run-id verdict field

## Task

Require WING read-only preflight operator decision records to carry the
machine-verifiable `event_log_run_id_matches_summary` verdict.

## Context

The artifact validator and review report already compute
`event_log_run_id_matches_summary: true` for clean candidates. The accepted
operator decision record still relied on checklist ids and hashes without
storing that exact machine verdict in the durable JSON.

## Constraints

- Do not contact WING or backend services.
- Do not add live authority or readiness closure.
- Keep the change scoped to the first MVP blocker.
- Preserve fail-closed behavior for older or incomplete decision records.

## Acceptance

- Review decision templates include `event_log_run_id_matches_summary: true`.
- The shared required-fields contract includes the verdict field.
- Accepted or rejected decision records without the true verdict are rejected.
- Runbook, no-WING surfaces, and focused tests reflect the new field.
