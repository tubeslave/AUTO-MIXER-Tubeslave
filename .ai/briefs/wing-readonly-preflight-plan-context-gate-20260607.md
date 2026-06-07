# Brief: WING Read-Only Preflight Plan Context Gate

## Task

Make the no-WING release check reject WING read-only preflight plan artifacts
that omit the current MVP blocker, manual-review boundary, fail-closed
conditions, or post-run review commands.

## Context

The first open MVP blocker is the real WING read-only connection proof. The
offline `--plan-only` preview is useful only if it is clearly tied to that
blocker and tells the operator what must happen after the real artifact is
captured. The runner now emits that context, but the release-check schema still
needs to enforce it.

## Safety Boundary

- Do not open WebSocket, query WING, or mutate any console state.
- Do not close any MVP blocker from an offline plan artifact.
- Keep live write, live authority, and readiness closure false.
- Treat missing manual-review or fail-closed context as a schema failure.

## Acceptance Criteria

- Plan artifact validation requires `selected_mvp_blocker` and blocker status.
- Plan artifact validation requires `operator_required` and
  `manual_review_required_before_readiness_update`.
- Plan artifact validation requires the expected fail-closed condition list.
- Plan artifact validation requires the artifact-review and decision-review
  commands to match the canonical no-WING command builders.
- Handoff packet validation requires the referenced plan artifact to pass the
  same offline schema gate before the packet is treated as ready.
- JSON and Markdown summaries expose the new fields.
