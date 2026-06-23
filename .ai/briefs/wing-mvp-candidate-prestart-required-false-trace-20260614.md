# Brief: WING MVP candidate prestart required-false trace

## Task

Make accepted-candidate WING read-only prestart traces carry the backend-port
required-false safety detail fields through evidence intake, artifact review,
decision review, and the shared prestart contract helper.

## Context

The WING read-only evidence intake now rejects stale or unsafe referenced
no-WING prestart detail, but downstream candidate review rows did not expose the
same required-false fields for operator and proof-plan inspection.

## Safety Boundary

- No WING hardware contact.
- No backend or WING listener start/stop.
- No OSC/WING writes.
- No live authority or readiness closure.

## Success Criteria

- Candidate review rows include the required-false prestart detail fields.
- The shared accepted-candidate prestart helper rejects missing or truthy fields.
- Artifact and decision review markdown render the fields.
- Focused WING evidence/review tests pass.
