# Brief: WING supervised gate expected-artifact marker

## Context

The accepted-candidate prestart contract now requires expected-artifact binding
before a WING read-only decision can become `accepted_decision_ready`. The final
handoff operator-sequence step still needs to name that binding explicitly so an
old three-marker gate cannot guide supervised proof planning.

## Task

Require the final supervised prestart gate to include
`expected-artifact-bound accepted source id` alongside `accepted_decision_ready`,
`valid accepted-candidate prestart trace`, and
`WING_SUPERVISED_FADER_GAIN_PROOF`.

## Acceptance

- The shared supervised prestart gate contract includes the new marker.
- no-WING handoff validation rejects old final steps that omit the new marker.
- Project Director, WING rails, runbook, and MVP docs surface the new marker.
- No backend, WING, OSC, or live-write behavior is added.
