# WING Read-Only Live Preflight

## Context

The physical WING Rack is reachable at `192.168.1.102` and Ableton is sending
multitrack audio into the console. The operator cockpit can connect to WING and
read cached console state, but live write tests must wait until the project can
also see audio meters and keep write gates closed by default.

## Goal

Add the smallest safe backend/frontend bridge for read-only audio monitoring:
start and stop audio capture for meters without starting automation or writing
to WING.

## Non-goals

- No fader, gain, EQ, compressor, routing, scene, or snapshot writes.
- No autonomous apply.
- No change to default operator mode.

## Verification

- WING OSC read-only connection succeeds.
- WING audio device exposes active multitrack channels.
- Dashboard can report audio capture as running and channels as active.
- Tests and frontend build pass.
