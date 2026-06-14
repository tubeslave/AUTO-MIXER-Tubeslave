# Brief: WING MVP Handoff Anchor Digest Review Step Index

## Task

Tighten the WING read-only handoff gate so the operator sequence proves that the
`mvp_blocker_anchor` and `mvp_blocker_anchor_sha256` review step stays at its
expected contract index before supervised proof planning.

## Context

The current first open MVP blocker is still WING read-only connection proof.
Recent slices validated the handoff operator sequence count and final supervised
prestart gate index. A packet could still move the anchor digest review step
earlier in the sequence while preserving a count of 10 and the final supervised
gate at index 9.

## Safety Boundary

This task is no-WING and read-only. It must not contact hardware, grant live
authority, execute tests from the handoff packet, or mark the blocker closed.

## Deliverable

Add a shared expected index for the anchor digest review step, fail closed on
index mismatch, expose the expected/match fields in no-WING summaries, and keep
Project Director/autoloop docs aligned.
