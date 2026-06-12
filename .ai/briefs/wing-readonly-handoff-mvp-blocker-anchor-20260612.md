# Brief: WING Read-Only Handoff MVP Blocker Anchor

Date: 2026-06-12

## Task

Bind the offline WING read-only preflight handoff packet to the current first
open MVP blocker so stale handoff packets cannot pass no-WING validation after
the blocker table changes.

## Context

The current first open blocker is `WING read-only connection proof`. The
handoff packet already proves read-only commands, backend-port prestart gates,
and supervised prestart gates, but the no-WING schema check did not compare the
packet to the current `.ai/project-layer/wing_mvp_blockers.md` first-open row.

## Safety Boundary

- No WING hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- No listener stop or reconfiguration.
- No readiness closure from offline evidence.

## Expected Outcome

Handoff packets include `mvp_blocker_anchor`; no-WING validation fails closed
when the anchor does not match the current first open blocker row.
