# Brief: WING Handoff Anchor Digest Operator Review Step

Date: 2026-06-12

## Task

Require the WING read-only handoff packet operator sequence to include a
dedicated review step for the current MVP blocker anchor and SHA256 digest.

## Context

The handoff packet already carries `mvp_blocker_anchor` and
`mvp_blocker_anchor_sha256`, and no-WING validation rejects stale or mismatched
digests. The operator sequence should also explicitly tell reviewers to compare
both fields before supervised proof planning.

## Safety Boundary

- No WING hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- No listener stop or reconfiguration.
- No readiness closure from offline evidence.

## Expected Outcome

Generated handoff packets contain a review step naming `mvp_blocker_anchor`,
`mvp_blocker_anchor_sha256`, and the current first open blocker. The no-WING
validator fails closed when that step is missing.
