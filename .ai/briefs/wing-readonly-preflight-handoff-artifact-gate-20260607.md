# Task: WING Read-Only Preflight Handoff Artifact Gate

## Context

The no-WING release check already builds an offline WING read-only preflight
plan and an operator handoff packet. The plan artifact is schema-validated, but
the handoff packet was only generated as text by the release check. That leaves
the operator sequence less machine-checkable than the plan itself.

## Goal

Make the release check validate the generated handoff packet as a structured
read-only operator gate. The handoff artifact must not contact WING, query a
console, execute tests, close readiness rows, or grant live authority.

## Safety Constraints

- No live WING, OSC, MIDI, TCP, or backend WebSocket activity.
- No mixer write command, proposal apply, rollback, mute, reset, or fader/gain
  send.
- The artifact must preserve the real hardware proof requirement.
- Fail closed if the handoff omits plan preview, capture, artifact review, or
  decision-review commands.

## Expected Output

- no-WING release check writes a JSON handoff artifact.
- no-WING release check validates the artifact schema after generation.
- Summary JSON/Markdown expose the handoff artifact gate.
- Tests cover valid, missing, and unsafe handoff artifacts.
