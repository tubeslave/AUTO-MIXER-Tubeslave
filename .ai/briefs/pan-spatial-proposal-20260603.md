# pan-spatial-proposal-20260603

## Context

The project goal explicitly requires automatic panning and console pan control.
The existing `auto_panner` modules contain useful heuristics but include apply
paths. Current live execution remains supervised fader/gain only.

## Scope

- Add a transport-free spatial panning recommendation module.
- Emit pan settings as proposal-only Product Layer payloads.
- Require stereo-pair, stage-position, center-anchor, and mono-compatibility
  evidence before creating proposals.
- Register the method in the processing registry and project-goal matrix.
- Add tests proving no live writes or queueable apply requests are created.

## Non-goals

- No live pan, WING, OSC, MIDI, or queue apply writes.
- No changes to legacy `auto_panner.py` behavior.
- No frontend panel work.
- No promotion beyond assisted/proposal-only.
