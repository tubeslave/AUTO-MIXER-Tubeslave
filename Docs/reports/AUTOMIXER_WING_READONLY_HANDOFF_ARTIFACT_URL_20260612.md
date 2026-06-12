# WING Read-Only Handoff Artifact URL

## Summary

Handoff packet artifact review now fails closed when the packet's top-level
`url` differs from the expected read-only backend URL.

## Safety Boundary

- No WING hardware contact.
- No backend WebSocket opened.
- No OSC writes.
- No live authority granted.
- Invalid handoff artifacts remain blocked.

## Validation

- Focused handoff artifact schema tests.
- WING/no-WING adjacent tests.
