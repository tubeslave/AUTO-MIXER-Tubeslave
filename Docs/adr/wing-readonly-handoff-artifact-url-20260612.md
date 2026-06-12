# ADR: WING read-only handoff artifact URL

## Context

The WING read-only handoff packet contains the backend URL used to generate
preview and capture commands. no-WING command validation checks the generated
handoff command, but artifact schema review did not reject a packet artifact
whose stored URL drifted from the expected read-only backend URL.

## Options Considered

1. Keep validating only the embedded preview/capture commands.
2. Validate the top-level handoff packet URL as well.

## Decision

Validate the top-level handoff packet URL.

## Why This Won

The handoff artifact is operator-facing evidence. If it records the wrong
backend URL, the artifact should fail before any backend/WING WebSocket session
is opened.

## Rejected Alternatives

Relying only on embedded command validation was rejected because the top-level
packet field can still mislead review summaries.

## Implementation Plan

- Add URL mismatch detection to handoff packet artifact review.
- Expose `url` and `artifact_root` in the review detail.
- Add tests for valid and invalid URL cases.

## Test Plan

- Focused handoff artifact schema tests.
- WING/no-WING adjacent tests.
- Full repository pytest.

## Risks And Rollback

The change is offline/read-only and does not contact WING, backend WebSocket,
or OSC. Rollback is to remove the URL validation and related assertions.
