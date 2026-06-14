# Brief: WING MVP Handoff Declared Director Self-Check

## Task

Make the WING read-only handoff packet declare the same Project Director
selected-action self-check fields that the no-WING gate recomputes.

## Context

The no-WING handoff summary already exposes expected, actual, and match-verdict
Project Director selected-action digest fields. The handoff packet itself still
only carried the shorter anchor/source/SHA subset, so the operator-facing packet
was less self-contained than the release gate summary.

## Safety Boundary

This is offline handoff metadata and validation work. It must not start a
backend, contact WING, query a console, write to a console, grant live
authority, update readiness, or close `WING read-only connection proof`.

## Deliverable

Add packet-declared Project Director expected/actual digest fields, render them
in the packet Markdown, make no-WING validation fail on stale declared
self-check fields, and cover the behavior with focused tests.
