# master-bus-optimization-proposal-20260603

## Context

The project goal requires automatic master bus optimization. Existing offline
renderers and quality scoring expose loudness and master-stage concepts, but
the current live runtime remains supervised fader/gain only. Main bus, limiter,
matrix, EQ, and compressor writes can affect the entire PA and are not approved
live surfaces.

## Scope

- Add a transport-free master bus recommendation module.
- Emit loudness, true-peak, ceiling, broad EQ, and glue-compression settings as
  proposal-only Product Layer payloads.
- Explicitly block unsafe loudness-up cases when true-peak headroom or approved
  limiter/dynamics proof is missing.
- Register the method in the processing registry, proposal-only contracts, and
  project-goal capability matrix.
- Add tests proving no live writes or queueable apply requests are created.

## Non-goals

- No live main bus, matrix, EQ, compressor, limiter, WING, OSC, MIDI, or queue
  apply writes.
- No changes to offline render/mastering signal chains.
- No frontend panel work.
- No promotion beyond assisted/proposal-only.
