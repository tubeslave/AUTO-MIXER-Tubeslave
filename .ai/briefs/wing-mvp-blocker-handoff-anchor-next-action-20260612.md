# Brief: WING MVP Blocker Handoff Anchor Next Action

Date: 2026-06-12

## Task

Synchronize the WING MVP blocker table next action with the new no-WING
handoff `mvp_blocker_anchor` validation requirement.

## Context

The no-WING handoff validator now rejects stale handoff packets unless
`mvp_blocker_anchor` matches the current first open MVP blocker row. The live
blocker table and generated progress report are the inputs Project Director uses
for the next task directive, so they must name the same requirement.

## Safety Boundary

- No WING hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- No listener stop or reconfiguration.
- No readiness closure from offline evidence.

## Expected Outcome

The first WING MVP blocker next action and generated latest progress report
require `mvp_blocker_anchor_matches_current_first_open: true` before supervised
proof planning.
