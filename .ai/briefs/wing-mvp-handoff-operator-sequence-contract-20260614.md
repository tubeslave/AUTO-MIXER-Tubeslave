# Brief: WING MVP Handoff Operator Sequence Contract

## Task

Move the WING read-only handoff operator sequence into the shared contract and
require no-WING validation to fail closed when the packet sequence drifts from
that contract.

## Context

The handoff packet declares a SHA256 digest over the actual operator sequence.
That detects post-generation edits, but it does not prove a regenerated packet
still matches the canonical ordered checklist expected by the supervised WING
MVP.

## Safety Boundary

This task is offline and read-only. It must not contact WING, execute packet
commands, grant live authority, or close the WING read-only connection proof
blocker.

## Deliverable

Define the expected handoff operator sequence in
`backend/wing_readonly_preflight_contract.py`, have the builder emit contract
digest/match fields, have no-WING validation recompute and fail closed on
contract drift, and update WING MVP rails, reports, and tests.
