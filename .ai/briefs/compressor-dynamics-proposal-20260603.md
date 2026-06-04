# compressor-dynamics-proposal-20260603

## Context

The project goal explicitly requires automatic compression and full console
compressor control. The current safe live executor is still supervised
fader/gain only, while legacy compressor code contains direct mixer behavior.

## Scope

- Add a transport-free compressor dynamics recommendation module.
- Emit compressor settings as proposal-only Product Layer payloads.
- Register the method in the processing registry and project-goal matrix.
- Add tests that prove no live writes or queueable apply requests are created.

## Non-goals

- No live compressor, dynamics, WING, OSC, MIDI, or queue apply writes.
- No changes to legacy `auto_compressor.py` behavior.
- No frontend panel work.
- No promotion beyond assisted/proposal-only.
