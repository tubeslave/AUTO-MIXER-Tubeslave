# Brief: Read-Surface Fixture Closure Plan Artifact Gate

## Task

Generate and validate the read-surface physical-readback fixture closure plan as
a no-WING artifact.

## Context

The fixture closure plan already exists as an in-memory read-only schema. It
turns accepted operator records into manual closure candidates while keeping
fixture updates and live authority blocked. The no-WING run should also produce
and validate the generated operator artifact so closure planning cannot drift
from the release summary.

## Safety Boundary

- Do not update fixture baselines automatically.
- Do not close physical readback gaps.
- Do not query or mutate WING, dLive, OSC, MIDI, TCP, or WebSocket transports.
- Do not grant live widening, live authority, or full autonomous readiness.
- Keep closure as a manual operator/repository step after accepted evidence.

## Acceptance Criteria

- Add an offline closure-plan builder with Markdown and JSON output.
- no-WING writes `read_surface_physical_readback_fixture_closure_plan.json`.
- no-WING validates the generated JSON artifact after the builder runs.
- Summary JSON and Markdown expose the artifact result separately from the
  in-memory schema result.
- Tests cover valid, missing, and unsafe closure-plan artifacts.
