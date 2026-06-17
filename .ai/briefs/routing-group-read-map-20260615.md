# Routing / Group Read-only Topology Map

Date: 2026-06-15
Status: implemented

## Task

Move Routing / Groups from `read_only_map_needed` to a concrete read-only map
artifact while preserving the current WING live-write boundary.

## Constraints

- Do not contact WING, OSC, MIDI, or mixer clients.
- Do not create route, group, DCA, matrix, or send writes.
- Use already captured cached/static Product Layer rows only.
- Surface freshness, destructive-change markers, group membership, and coverage
  blockers must be explicit.
- The map must not be queueable and must not grant live authority.

## Acceptance

- `routing_group_read_map` produces a read-only topology artifact.
- Missing/stale surfaces, missing routing destructive markers, and missing
  group/DCA membership keep the map incomplete.
- Registry, capability matrix, backlog, ADR, report, and tests reflect the
  boundary.
