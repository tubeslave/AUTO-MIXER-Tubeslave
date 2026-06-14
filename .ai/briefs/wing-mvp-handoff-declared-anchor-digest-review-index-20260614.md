# Brief: WING MVP Handoff Declared Anchor Digest Review Index

## Task

Make the packet-declared `mvp_blocker_anchor_digest_review_step_index` part of
the no-WING handoff validation contract.

## Context

The current WING read-only handoff gate validates the actual operator sequence
anchor digest review step index. The generated handoff packet also carries a
top-level declared index, but no-WING validation did not yet compare that field
against the shared contract or the actual operator sequence.

## Safety Boundary

This task is offline and read-only. It must not contact WING, execute packet
commands, grant live authority, or close the WING read-only connection proof
blocker.

## Deliverable

Fail closed when the declared anchor digest review step index mismatches the
shared contract or the actual operator sequence, expose the declared match
fields in summaries, and keep WING MVP docs and Project Director rails aligned.
