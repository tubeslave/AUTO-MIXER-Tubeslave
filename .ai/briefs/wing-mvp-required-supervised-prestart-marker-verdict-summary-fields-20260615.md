# Brief: WING MVP Required Supervised Prestart Marker Verdict Summary Fields

## Task

Expose top-level verdicts proving `required_supervised_prestart_gate_markers`
matches the supervised proof marker contract and the structured supervised gate
marker list.

## Context

The no-WING handoff summary already exposes the top-level required marker list,
its count, and structured supervised gate marker verdicts. The top-level marker
list itself still lacked direct match verdicts. Operators and Project Director
should not infer this from adjacent structured-gate fields.

## Scope

- Add top-level no-WING fields:
  `required_supervised_prestart_gate_markers_match_contract` and
  `required_supervised_prestart_gate_markers_match_supervised_gate`.
- Render them in JSON and Markdown summaries.
- Add Project Director baseline markers and tests.

## Out Of Scope

- Contacting WING hardware.
- Granting live authority.
- Closing readiness from offline evidence.
