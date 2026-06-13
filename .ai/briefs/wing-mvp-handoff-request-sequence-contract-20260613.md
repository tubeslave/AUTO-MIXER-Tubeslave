# WING MVP Handoff Request Sequence Contract

## Task

Expose the read-only WING preflight request sequence in the operator handoff
packet and make no-WING validation reject packet/plan drift.

## Context

The supervised WING MVP still waits on real WING read-only evidence and manual
review. The handoff operator sequence asks the operator to confirm that the
request sequence is read-only, but the packet did not carry the exact expected
WebSocket request sequence for inspection.

## Safety Boundary

This is offline/no-WING contract hardening. It does not contact hardware,
grant live authority, close readiness, or allow mixer writes.

## Expected Outcome

- Handoff packets include `request_sequence` and `request_count`.
- Handoff markdown renders the expected request sequence for operator review.
- no-WING handoff validation fails closed when the packet sequence drifts from
  the generated plan artifact or expected sequence.
- Focused tests and the standard suite pass before commit.
