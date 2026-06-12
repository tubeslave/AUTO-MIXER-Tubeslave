# Brief: WING MVP Blocker Handoff Anchor Digest Next Action

Date: 2026-06-12

## Task

Synchronize the WING MVP blocker table next action with the no-WING handoff
`mvp_blocker_anchor_sha256` validation requirement.

## Context

The no-WING handoff validator now recalculates the first-open blocker anchor
digest and fails closed unless the handoff summary reports
`mvp_blocker_anchor_sha256_matches_current_first_open: true`. The live blocker
table and generated progress report are the inputs Project Director uses for
the next autonomous task directive, so they must name the digest requirement.

## Safety Boundary

- No WING hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- No listener stop or reconfiguration.
- No readiness closure from offline evidence.

## Expected Outcome

The first WING MVP blocker next action and generated latest progress report
require both the handoff anchor and the anchor SHA256 digest to match the
current first open blocker before supervised proof planning.
