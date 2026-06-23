# Brief: WING MVP review checklist run-id binding

## Task

Require the operator review checklist for WING read-only preflight decisions to
explicitly confirm event-log `run_id` binding.

## Context

The artifact validator and review report now verify that every `events.jsonl`
row has the same `run_id` as `summary.json`. The accepted-decision checklist
still confirmed event-log read-only sequence but did not name row-level run-id
binding as a required operator confirmation.

## Constraints

- Do not contact WING or backend services.
- Do not add live authority or readiness closure.
- Keep the change scoped to the first MVP blocker.
- Preserve the existing review-decision ledger and fail-closed behavior.

## Acceptance

- `WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS` includes
  `confirm_event_log_run_id_binding`.
- The review tool renders the new checklist item and templates include it.
- Old accepted-decision JSON without the new checklist id is rejected.
- Handoff/no-WING surfaces expose the expanded required checklist.
