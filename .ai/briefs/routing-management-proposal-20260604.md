# routing-management-proposal-20260604

## Context

The project goal requires console routing control. Current live execution
remains supervised fader/gain only, while routing, snapshots, and scene writes
can rewrite broad console state. Existing routing write paths remain outside
the approved live executor.

## Scope

- Add a transport-free routing management recommendation module.
- Emit route-diff settings as proposal-only Product Layer payloads.
- Use current/proposed route endpoints, route readback, rollback baseline,
  operator intent, active-performance guard, current-route activity, destructive
  scope, source confidence, route confidence, and persistence evidence.
- Register the method in the processing registry, proposal-only contracts, and
  project-goal capability matrix.
- Add tests proving no live writes or queueable apply requests are created.

## Non-goals

- No live routing, snapshot, scene, WING, OSC, MIDI, or queue apply writes.
- No changes to existing routing handlers or mixer-client route behavior.
- No frontend panel work.
- No promotion beyond assisted/proposal-only.
