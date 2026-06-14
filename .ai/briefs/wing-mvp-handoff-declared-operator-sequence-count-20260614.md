# Brief: WING MVP Handoff Declared Operator Sequence Count

## Task

Make the packet-declared operator sequence count part of the no-WING handoff
validation contract.

## Context

The WING handoff validator computes `operator_sequence_count` from the actual
operator sequence. The generated packet did not yet carry a top-level declared
count, so operator-facing metadata could not be checked against the actual
sequence length before WING read-only proof work.

## Safety Boundary

This task is offline and read-only. It must not contact WING, execute packet
commands, grant live authority, or close the WING read-only connection proof
blocker.

## Deliverable

Add packet-declared operator sequence count fields, fail closed when they drift
from the actual sequence or shared contract, expose the declared match fields in
summaries, and keep WING MVP docs and Project Director rails aligned.
