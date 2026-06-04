# mute-management-proposal-20260603

## Context

The project goal requires console mute control and real-time corrections. Current
live execution remains supervised fader/gain only, while mute writes can create
dropouts, silence critical sources, or hide routing/DCA mistakes. Existing
legacy and voice-control mute write paths remain outside the approved live
executor.

## Scope

- Add a transport-free mute management recommendation module.
- Emit channel mute/unmute settings as proposal-only Product Layer payloads.
- Use source role, current mute state, auto-muted provenance, signal activity,
  idle duration, operator intent, feedback risk, dropout risk, source
  confidence, and persistence evidence.
- Register the method in the processing registry, proposal-only contracts, and
  project-goal capability matrix.
- Add tests proving no live writes or queueable apply requests are created.

## Non-goals

- No live mute, WING, OSC, MIDI, voice, or queue apply writes.
- No changes to legacy voice-control or mixer-client mute behavior.
- No frontend panel work.
- No promotion beyond assisted/proposal-only.
