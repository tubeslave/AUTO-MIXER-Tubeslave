# WING Read-Only Handoff URL and Artifact Root

## Summary

The no-WING release handoff command now explicitly includes the read-only
backend URL and artifact root, and no-WING evidence reports whether those
handoff fields match the capture command.

## Safety Boundary

- No WING hardware contact.
- No backend WebSocket opened.
- No OSC writes.
- No live authority granted.
- The change remains offline/read-only and fails closed on handoff drift.

## Validation

- Focused no-WING release-check tests.
- WING MVP rails tests.
