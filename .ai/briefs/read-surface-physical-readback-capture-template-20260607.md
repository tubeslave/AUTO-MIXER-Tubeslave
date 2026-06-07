# Brief: Read-Surface Physical Readback Capture Template

## Task

Add an offline capture-template artifact for the next read-surface physical
readback target.

## Context

The current no-WING chain can generate a capture packet, review queue,
acceptance template, and fixture-closure plan. The packet names the remaining
targets, but operators still need a structured pending artifact shape to fill
with real physical readback data before intake and review.

## Safety Boundary

- Do not query or mutate a mixer.
- Do not treat the template as physical proof.
- Do not accept artifacts, update fixtures, close read-surface gaps, or widen
  live authority from the template.
- Keep execution, fixture-closure, live authority, and full autonomous flags
  false at both template and record level.

## Acceptance Criteria

- A tool builds a pending capture template from the capture packet.
- The default template targets the next physical readback surface and can select
  a specific surface or all remaining targets.
- no-WING generates and validates the template before the artifact-review queue.
- Validation rejects filled proof payloads or authority claims disguised as a
  template.
- Focused tests cover builder behavior, unsafe artifacts, command order, and
  summary output.
