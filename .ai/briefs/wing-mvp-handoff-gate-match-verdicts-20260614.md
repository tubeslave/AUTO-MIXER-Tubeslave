# Brief: WING MVP Handoff Gate Match Verdicts

## Task

Make the WING read-only handoff packet and no-WING summary expose named match
verdicts for backend and supervised prestart gates.

## Context

The validator already compared `backend_port_prestart_gate` with
`expected_backend_port_prestart_gate` and `supervised_prestart_gate` with
`expected_supervised_prestart_gate`. The operator-facing packet and no-WING
summary still relied on object equality and a generic Markdown
`actual_matches_expected` label rather than explicit machine-readable verdict
fields.

## Safety Boundary

This is offline handoff metadata and validation work. It must not start a
backend, contact WING, query a console, write to a console, grant live
authority, update readiness, or close `WING read-only connection proof`.

## Deliverable

Add named gate match fields to the handoff packet, validate stale declarations
in no-WING, render the verdicts in summaries, update Project Director rail
markers, and cover the behavior with focused tests.
