# Brief: WING MVP Handoff Director Action Digest

## Task

Bind the WING read-only handoff packet to the current Project Director selected
action digest.

## Context

The active blocker remains `WING read-only connection proof`. Project Director
now emits `selected_action_sha256` and current-first-command baseline coverage
fields. The handoff packet already carries the first-open-blocker anchor and
operator-sequence digest, but it did not yet prove that the packet was built
against the same selected action Project Director is directing Codex to follow.

## Safety Boundary

This task is offline and read-only. It must not contact WING, execute packet
commands, grant live authority, update readiness, or close the WING read-only
connection proof blocker.

## Deliverable

Add a Project Director selected-action anchor to the handoff packet, validate
the anchor and digest in no-WING release checks, expose the fields in JSON and
Markdown summaries, and add regression coverage for digest drift.
