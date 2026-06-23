# Brief: WING MVP Supervised Prestart Step Index Summary Fields

## Task

Expose the supervised prestart gate's required/final-step contract as top-level
no-WING summary fields.

## Context

The structured `supervised_prestart_gate` already records that the gate is
required, belongs at operator sequence step index `9`, and must be final before
the supervised fader/gain proof. Operators and automation should be able to
verify these values without parsing the nested object.

## Scope

- Add top-level no-WING fields for:
  `supervised_prestart_gate_required`,
  `supervised_prestart_gate_operator_sequence_step_index`, and
  `supervised_prestart_gate_operator_sequence_step_must_be_final`.
- Mirror the fields in JSON and Markdown summaries.
- Add Project Director baseline markers.

## Out Of Scope

- Contacting WING hardware.
- Granting live authority.
- Closing readiness from offline evidence.
