# Brief: WING MVP Current Command Director Action Digest

## Task

Make the Project Director `Current First Command` require the handoff packet's
Project Director selected-action digest fields.

## Context

The WING handoff packet now emits and validates
`project_director_selected_action_*` fields. The current Project Director action
still needs to mention those fields explicitly so future Goal-mode cycles do
not treat the new digest proof as optional.

## Safety Boundary

This task is offline planning and validation sync. It must not contact WING,
execute packet commands, grant live authority, update readiness, or close the
`WING read-only connection proof` blocker.

## Deliverable

Add the handoff director digest fields to `Current First Command`, baseline
coverage markers, autoloop docs, and rails tests. Regenerate the latest MVP
progress report and record validation.
