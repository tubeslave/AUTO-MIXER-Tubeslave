# Brief: mixing-station-trim-read-parity-20260605

## Context

The supervised fader/gain gate now relies on canonical cached addresses and
fresh `state_timestamps`. WING, dLive, offline validation, recommendation
bridges, and the manual gate use `/ch/{N}/in/set/trim` for input trim. Mixing
Station still mapped its public gain helper through `/ch/{N}/gain`, so cache
pre-read and rollback evidence could miss the canonical trim key.

## Task

Make Mixing Station use the same canonical fader/gain read contract as the rest
of the backend. Input gain/trim updates should populate
`/ch/{N}/in/set/trim`, write helpers should send the equivalent Mixing Station
preamp gain address, and read helpers should query through the same key.

## Safety Boundaries

- No live Mixing Station, WING, or dLive traffic during implementation.
- No new write authority.
- Supervised writes remain limited to existing `fader` and `gain` gates.
- EQ, dynamics, routing, FX, scenes, and presets remain quarantined.

## Expected Evidence

- Mixing Station `ch.N.preamp.gain` messages update canonical trim cache.
- `set_channel_gain()` sends `ch.N.preamp.gain` from `/ch/N/in/set/trim`.
- `query_parameter()` returns readback-compatible payloads for trim aliases.
- Tests use dummy sockets only.
