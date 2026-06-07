# Brief: Read-Surface Capture Packet Artifact Gate

## Task

Validate the generated read-surface physical-readback capture packet artifact in
the no-WING release check.

## Context

The capture packet builder renders the next operator-required physical-readback
target for broad read-surface coverage. The no-WING release check already runs
the builder, but the generated packet artifact needs to be read back as JSON and
validated before later operator gates can pass.

## Safety Boundary

- Do not contact WING, dLive, OSC, MIDI, TCP, WebSocket, or mixer hardware.
- Do not add live writes, fixture closure, readiness updates, or live authority.
- Keep the packet operator-required and non-executing.
- Fail closed if the generated artifact claims console queries, mutations,
  execution permission, live widening, or autonomous readiness.

## Acceptance Criteria

- no-WING writes `read_surface_physical_readback_capture_packet.json`.
- no-WING validates that generated JSON immediately after the builder runs.
- Summary JSON and Markdown expose the artifact gate result.
- Tests cover valid, missing, and unsafe capture-packet artifacts.
- Docs record that this is an offline operator gate, not physical proof.
