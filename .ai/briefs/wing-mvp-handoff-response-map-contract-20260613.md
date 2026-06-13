# WING MVP Handoff Response Map Contract

## Task

Expose the read-only WING preflight request-to-response mapping in the operator
handoff packet and make no-WING validation reject packet/plan drift.

## Context

The supervised WING MVP still waits on real WING read-only evidence and manual
review. The offline plan now exposes `expected_response_types_by_request_type`,
but the handoff packet did not carry the same map, leaving the operator-facing
packet less inspectable than the plan it references.

## Safety Boundary

This is a no-WING contract hardening step. It does not contact hardware, grant
live authority, close readiness, or allow mixer writes.

## Expected Outcome

- Handoff packets include `expected_response_types_by_request_type`.
- Handoff markdown renders the expected response map for operator review.
- no-WING handoff validation fails closed when the packet response map drifts
  from the shared contract or generated plan artifact.
- Focused tests and the standard suite pass before commit.
