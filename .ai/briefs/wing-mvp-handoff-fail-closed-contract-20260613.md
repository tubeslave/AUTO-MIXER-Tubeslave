# WING MVP Handoff Fail-Closed Contract

## Task

Expose the read-only WING preflight fail-closed conditions in the operator
handoff packet and make no-WING validation reject packet/plan drift.

## Context

The supervised WING MVP still waits on real WING read-only evidence and manual
review. The offline preflight plan already lists the conditions that must stop
the run safely, but the handoff packet did not carry the same list for operator
inspection.

## Safety Boundary

This is offline/no-WING contract hardening. It does not contact hardware,
grant live authority, close readiness, or allow mixer writes.

## Expected Outcome

- Handoff packets include `fail_closed_conditions`.
- Handoff markdown renders the fail-closed list for operator review.
- no-WING handoff validation fails closed when the packet list drifts from the
  shared contract or generated plan artifact.
- Focused tests and the standard suite pass before commit.
