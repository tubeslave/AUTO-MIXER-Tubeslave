# Task: Read-Surface Acceptance Template Artifact Gate

## Context

The no-WING release check builds a JSON acceptance-record template for physical
readback review candidates. Until this slice, that generated artifact was not
read back and schema-validated by the release check.

## Goal

Validate the generated acceptance-template artifact as a read-only operator
gate. The artifact must not claim fixture closure, live authority, execution
permission, or autonomous readiness.

## Safety Constraints

- No WING, dLive, OSC, MIDI, TCP, backend WebSocket, or console query.
- No acceptance template may close a fixture gap.
- Pending records must keep operator decision/reviewer fields as placeholders.
- Any authority, live-write, execution, or fixture-closure claim must fail the
  release check.

## Expected Output

- no-WING release check validates the generated acceptance-template JSON.
- no-WING summary exposes the artifact status and safety flags.
- Tests cover valid, missing, and unsafe template artifacts.
