# Brief: WING MVP read-only event log run-id binding

## Task

Bind every real WING read-only preflight `events.jsonl` row to the preflight
`run_id` recorded in `summary.json`.

## Context

The current real preflight review validates request ordering, response types,
readback identity, artifact paths, and SHA256 bindings. The event log rows
themselves need an explicit run-id binding so a copied event log from another
run cannot look like part of the reviewed physical artifact.

## Constraints

- Do not contact WING or backend services in tests.
- Do not add live authority or readiness closure.
- Keep the change scoped to read-only artifact integrity.
- Preserve the existing expected-plan and prestart-summary gates.

## Acceptance

- The runner writes `run_id` into every `events.jsonl` row.
- The evidence validator fails closed when any event row has a missing or
  mismatched `run_id`.
- The artifact review report exposes `event_log_run_id_matches_summary`.
- Focused runner/evidence/review tests cover both passing and mismatched
  event-log run ids.
