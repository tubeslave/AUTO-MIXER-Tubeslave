# bus-group-balance-proposal-20260603

## Context

The project goal requires automatic mix balance and control of DCA, groups, and
matrices. Current live execution remains supervised fader/gain only, while
group/DCA/matrix writes affect broad console state and have no approved proof
path. The project canon also requires Ayaic level planes to be checked at both
source and summed stem/group levels.

## Scope

- Add a transport-free bus/group balance recommendation module.
- Emit stem, group, DCA, matrix, and FX-return balance settings as
  proposal-only Product Layer payloads.
- Use Ayaic target planes, summed child loudness, assignment confidence,
  current fader readback, true-peak headroom, main-bus headroom, feedback risk,
  and persistence evidence.
- Register the method in the processing registry, proposal-only contracts, and
  project-goal capability matrix.
- Add tests proving no live writes or queueable apply requests are created.

## Non-goals

- No live DCA, group, matrix, bus, WING, OSC, MIDI, or queue apply writes.
- No changes to existing `autofoh_safety` DCA action behavior.
- No frontend panel work.
- No promotion beyond assisted/proposal-only.
