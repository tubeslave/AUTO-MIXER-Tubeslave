# fx-send-proposal-20260603

## Context

The project goal explicitly requires automatic effects management and effect
send tuning. Existing `auto_fx` contains a useful planner but also has an
`apply_to_mixer` path. Current live execution remains supervised fader/gain
only, and FX writes are not approved.

## Scope

- Add a transport-free FX send recommendation module.
- Emit FX send/return settings as proposal-only Product Layer payloads.
- Use source role, tempo, return headroom, feedback risk, intelligibility risk,
  contribution, and persistence evidence.
- Register the method in the processing registry and project-goal matrix.
- Add tests proving no live writes or queueable apply requests are created.

## Non-goals

- No live FX, send, insert, WING, OSC, MIDI, or queue apply writes.
- No changes to legacy `auto_fx.py` behavior.
- No frontend panel work.
- No promotion beyond assisted/proposal-only.
