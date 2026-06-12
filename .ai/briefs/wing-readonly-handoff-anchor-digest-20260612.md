# Brief: WING Read-Only Handoff Anchor Digest

Date: 2026-06-12

## Task

Add a stable digest to the WING read-only handoff `mvp_blocker_anchor` so the
offline packet can prove the displayed blocker-row anchor has not drifted or
been edited.

## Context

The handoff packet now includes `mvp_blocker_anchor` for the current first open
WING MVP blocker. The no-WING validator compares that object to the live blocker
table. A digest makes the anchor easier to inspect in summaries and gives
validators a compact contract to reject stale or tampered packets.

## Safety Boundary

- No WING hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- No listener stop or reconfiguration.
- No readiness closure from offline evidence.

## Expected Outcome

Handoff packets include `mvp_blocker_anchor_sha256`, and no-WING validation
fails closed unless the digest matches both the packet anchor and the current
first open blocker row.
