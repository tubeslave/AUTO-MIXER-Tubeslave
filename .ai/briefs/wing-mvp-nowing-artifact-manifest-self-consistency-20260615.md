# Brief: WING MVP no-WING Artifact Manifest Self-Consistency

## Task

Make the no-WING `artifact-manifest.json` prove its own required artifact IDs,
paths, SHA fields, and safety-denial fields are internally consistent with the
actual `artifacts[]` entries it writes.

## Context

The current MVP blocker remains `WING read-only connection proof`. The no-WING
bundle prepares an operator handoff for real WING read-only evidence, but the
bundle itself must stay offline and must not grant readiness or live authority.
The detached manifest already lists required artifacts and SHA values; it should
also expose an aggregate self-consistency verdict for operator review.

## Scope

- Add self-consistency verdict fields to `artifact-manifest.json`.
- Include the required artifact SHA field list in the detached manifest.
- Fail the self-consistency verdict when artifact rows drift from top-level
  declarations or safety-denial fields.
- Keep the change offline and no-WING only.

## Non-Goals

- Do not contact WING.
- Do not query or mutate mixer state.
- Do not close readiness.
- Do not grant live authority.

## Validation

- Targeted no-WING manifest tests.
- Focused WING/no-WING/Director tests.
- Full standard pytest before merge.
