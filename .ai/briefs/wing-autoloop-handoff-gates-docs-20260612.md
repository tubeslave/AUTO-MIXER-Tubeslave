# Brief: WING Autoloop Handoff Gates Docs

Date: 2026-06-12

## Task

Update Codex autoloop prompt and README so automated cycles carry the current
WING handoff gate fields before supervised proof planning.

## Context

Project Director selected action now requires blocker-anchor digest review and
supervised prestart trace gates. The autoloop prompt and README still named
only the older backend/supervised gate parity fields, so future automated
cycles could start from stale instructions.

## Safety Boundary

- No WING hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- No listener stop or reconfiguration.
- No readiness closure from offline evidence.

## Expected Outcome

Autoloop prompt, README, and tests require the same blocker-anchor digest review
and supervised trace gate fields as the current WING MVP blocker next action.
