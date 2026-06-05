# Brief: cache-freshness-read-dto-20260605

## Context

Read-only Product Layer snapshots now expose broad cached console state, but
cached values can be stale. Autonomous mixing cannot safely promote readback or
proposal decisions unless the system can distinguish fresh cache from unknown or
old data.

## Task

Add a read-only cache freshness contract for base WING channel/control state,
channel processing state, and dLive cached MIDI/TCP state. The work must not
query mixers, scan channels, send OSC/MIDI, or widen apply authority.

## Safety Boundaries

- No live OSC or dLive MIDI/TCP writes.
- No Product Layer calls to scan/query helpers.
- `fresh` requires timestamp evidence; missing timestamps must remain
  `unknown`, `missing`, `partial`, or `stale`.
- This does not make any EQ, dynamics, routing, scene, FX, or preset proposal
  live-applyable.

## Expected Evidence

- WING and dLive clients timestamp cache updates.
- Product Layer snapshots expose `cache_freshness` without transport calls.
- Focused tests cover fresh, stale, and missing cache states.
- Inventory still marks incomplete parity/proof blockers.
