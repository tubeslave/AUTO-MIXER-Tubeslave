# Brief: Read-Surface Artifact Review Queue Artifact Gate

## Task

Generate and validate the read-surface physical-readback artifact review queue as
a no-WING artifact.

## Context

The artifact review queue already exists as an in-memory read-only schema. It
collects physical-readback intake outputs before operators can accept or reject
candidate artifacts. The no-WING run should also produce and validate the
generated queue artifact so acceptance-template generation cannot skip a stale
or unsafe review boundary.

## Safety Boundary

- Do not accept, reject, or close physical readback fixture candidates.
- Do not query or mutate WING, dLive, OSC, MIDI, TCP, or WebSocket transports.
- Do not grant live widening, live authority, or full autonomous readiness.
- Keep queue review as a manual operator step before acceptance records.

## Acceptance Criteria

- Add an offline review-queue builder with Markdown and JSON output.
- no-WING writes `read_surface_physical_readback_artifact_review_queue.json`.
- no-WING validates the generated JSON artifact after the builder runs.
- Summary JSON and Markdown expose the artifact result separately from the
  in-memory schema result.
- Tests cover valid, missing, unsafe, ordering, CLI, and summary cases.
