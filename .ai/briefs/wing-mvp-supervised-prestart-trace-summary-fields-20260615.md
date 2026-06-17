# Brief: WING MVP Supervised Prestart Trace Summary Fields

## Task

Expose all supervised prestart trace prerequisites as top-level no-WING summary
fields.

## Context

The structured `supervised_prestart_gate` already requires accepted-candidate
prestart, artifact-root, and expected-artifact traces before any supervised
fader/gain proof. The expected-artifact requirement already has a flat summary
field; the sibling prestart and artifact-root requirements should be equally
grep-friendly for operator and automation review.

## Scope

- Add top-level no-WING fields for:
  `supervised_prestart_gate_requires_accepted_candidate_prestart_trace` and
  `supervised_prestart_gate_requires_accepted_candidate_artifact_root_trace`.
- Keep the existing expected-artifact flat field.
- Mirror all three fields in JSON and Markdown summaries.
- Require the new markers in Project Director Current First Command coverage.

## Out Of Scope

- Contacting WING hardware.
- Granting live authority.
- Closing readiness based on offline evidence.
