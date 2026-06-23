# Brief: WING MVP Supervised Prestart Step Text Summary Fields

## Task

Expose the supervised prestart gate operator-step text and a match verdict as
top-level no-WING summary fields.

## Context

The operator sequence already has a final supervised prestart step, and the
structured `supervised_prestart_gate` carries the same step text. Operators
should see a direct summary verdict that these strings match before supervised
fader/gain proof planning.

## Scope

- Add top-level no-WING fields:
  `supervised_prestart_gate_operator_sequence_step` and
  `supervised_prestart_gate_operator_sequence_step_matches_summary`.
- Render them in JSON and Markdown summaries.
- Add Project Director baseline markers.

## Out Of Scope

- Contacting WING hardware.
- Granting live authority.
- Closing readiness from offline evidence.
