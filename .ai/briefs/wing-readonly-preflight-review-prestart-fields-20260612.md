# Brief: WING Read-Only Preflight Review Prestart Fields

## Task

Make the artifact review report surface the backend-port prestart contract from
validated WING read-only preflight summaries.

## Context

The real `wing_readonly_preflight.v1` summary now carries prestart contract
fields and evidence intake validates them. The artifact review report should
show those fields to the operator during manual review instead of hiding them in
the raw summary.

## Constraints

- Do not contact WING hardware.
- Do not open a backend WebSocket WING session.
- Do not add or widen live-write authority.
- Keep the change to evidence metadata propagation, review rendering, and tests.

## Acceptance

- Evidence artifact records carry the prestart contract fields.
- Artifact review rows expose those fields.
- Review markdown renders the prestart command and failure detail fields.
