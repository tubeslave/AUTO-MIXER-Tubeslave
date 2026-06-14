# Brief: WING MVP Handoff Operator Sequence SHA256

## Task

Bind the WING read-only handoff packet's operator sequence to a SHA256 digest
and require no-WING validation to fail closed when the declared digest drifts
from the actual sequence.

## Context

The handoff packet now declares and validates its operator sequence count, and
the no-WING gate checks key step indexes and markers. It still did not bind the
entire ordered operator checklist, so an intermediate step could drift while
the count and required marker checks stayed green.

## Safety Boundary

This task is offline and read-only. It must not contact WING, execute packet
commands, grant live authority, or close the WING read-only connection proof
blocker.

## Deliverable

Add `operator_sequence_sha256` to the handoff packet, recompute it in the
no-WING handoff validator, fail closed on mismatch, expose the match result in
summaries, and align WING MVP docs, Project Director rails, and tests.
