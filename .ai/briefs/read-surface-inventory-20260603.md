# read-surface-inventory-20260603

## Context

The project goal requires the Automixer to read all available console
parameters and analyze channels, groups, matrices, buses, and effects before
autonomous control can be credible. Current code has many cached reads and
query helpers, but no explicit read-surface inventory.

## Scope

- Add a read-only console surface inventory.
- Cover channels, channel processing, buses/groups/main/DCA, matrices, FX,
  routing/sends, snapshots/scenes, user presets, and dLive.
- Mark partial and missing surfaces honestly.
- Connect the inventory to the project-goal capability map.

## Non-goals

- No mixer connection.
- No OSC/MIDI/network reads during tests.
- No live writes or readback promotion.
