# Brief: WING MVP Next Manual Artifact Required By Packet Next Action

## Context

The active MVP blocker remains `WING read-only connection proof`. The no-WING
handoff packet is offline operator guidance and must not be mistaken for a real
WING proof artifact.

## Task

Promote `next_manual_artifact_required: real wing_readonly_preflight.v1
summary.json` into the WING MVP rails for the current read-only handoff
boundary.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure backend/WING listeners.
- Do not add live OSC/WING writes.
- Keep the change limited to offline tooling, docs, rails, council artifacts,
  and regression tests.

## Validation

- Focused handoff packet, no-WING release, Project Director, and WING MVP rail
  tests.
- Project Director selected action still targets `WING read-only connection
  proof`.
- Direction guard remains `on_target`.
- Standard backend pytest before commit.
