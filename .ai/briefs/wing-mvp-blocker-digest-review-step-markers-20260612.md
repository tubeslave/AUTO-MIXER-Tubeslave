# Brief: WING MVP Blocker Digest Review Step Markers

## Context

The read-only preflight runbook and handoff validator require the
`operator_sequence_mvp_blocker_anchor_digest_review_step` to name
`mvp_blocker_anchor`, `mvp_blocker_anchor_sha256`, and the `current first open
blocker`. The Project Director selected action only required the boolean
`operator_sequence_has_mvp_blocker_anchor_digest_review: true`.

## Task

Promote the digest-review step marker requirement into the first WING MVP
blocker selected action, generated progress, and autoloop docs.

## Safety Boundary

This is documentation and test synchronization only. It does not contact WING,
does not open a backend/WING session, and does not grant live write authority.
