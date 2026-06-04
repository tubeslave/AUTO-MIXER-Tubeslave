# deesser-proposal-layer-20260603

## Context

The project goal explicitly requires automatic de-essing. The capability matrix
identified de-essing as missing: the repo had AI/rule vocabulary for
`adjust_deesser` and sibilance detectors, but no registered transport-free
de-esser method or proposal-only contract.

## Scope

- Add a vocal de-esser recommendation module.
- Keep it transport-free and proposal-only.
- Register it in the processing registry and project-goal matrix.
- Add tests that prove no live writes or queueable apply requests are created.

## Non-goals

- No live de-esser, compressor, dynamic-EQ, EQ, OSC, MIDI, or WING writes.
- No frontend panel work.
- No promotion beyond proposal-only.
