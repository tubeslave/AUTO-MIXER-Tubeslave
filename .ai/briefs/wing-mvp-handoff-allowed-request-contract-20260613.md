# WING MVP Handoff Allowed Request Contract

## Task

Expose the read-only WING preflight allowed request types and request-id policy
in the operator handoff packet and make no-WING validation reject packet/plan
drift.

## Context

The supervised WING MVP still waits on real WING read-only evidence and manual
review. The offline preflight plan constrains backend WebSocket traffic through
`allowed_request_types` and `request_id_policy`, but the handoff packet did not
carry the same contract fields for operator inspection.

## Safety Boundary

This is offline/no-WING contract hardening. It does not contact hardware,
grant live authority, close readiness, or allow mixer writes.

## Expected Outcome

- Handoff packets include `allowed_request_types` and `request_id_policy`.
- Handoff markdown renders both fields for operator review.
- no-WING handoff validation fails closed when those fields drift from the
  shared contract or generated plan artifact.
- Focused tests and the standard suite pass before commit.
