# Brief: WING MVP Supervised Prestart Marker Verdict Summary Fields

## Task

Expose verdicts that the supervised prestart gate required markers match the
contract and the top-level summary marker list.

## Context

The no-WING handoff summary already shows `required_supervised_prestart_gate_markers`
and the structured `supervised_prestart_gate.required_markers`. Operators should
also see direct pass/fail fields proving those markers still match the expected
contract before supervised fader/gain proof planning.

## Scope

- Add top-level no-WING fields:
  `supervised_prestart_gate_required_markers`,
  `supervised_prestart_gate_required_markers_match_contract`, and
  `supervised_prestart_gate_required_markers_match_summary`.
- Render them in JSON and Markdown summaries.
- Add Project Director baseline markers.

## Out Of Scope

- Contacting WING hardware.
- Granting live authority.
- Closing readiness from offline evidence.
