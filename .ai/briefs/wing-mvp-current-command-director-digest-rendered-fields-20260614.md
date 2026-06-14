# Brief: WING MVP Current Command Director Digest Rendered Fields

## Task

Make the `Current First Command` baseline require the full rendered Project
Director digest field set that the no-WING handoff summary already emits.

## Context

The current command already names the expected/actual selected-action digest
fields, but the baseline marker tuple and autoloop docs only required the
shorter digest set. That left room for a future Goal-mode cycle to drop
`expected_*`, `actual_*`, or `*_available_matches_digest` fields while Project
Director still treated the command as covered.

## Safety Boundary

This is offline planning, documentation, and test coverage work. It must not
start a backend, contact WING, query a console, write to a console, grant live
authority, update readiness, or close `WING read-only connection proof`.

## Deliverable

Add the rendered expected/actual digest fields and coverage marker-count field
to the baseline marker set, current command docs, autoloop docs, and rails
tests. Regenerate latest MVP progress and record validation.
