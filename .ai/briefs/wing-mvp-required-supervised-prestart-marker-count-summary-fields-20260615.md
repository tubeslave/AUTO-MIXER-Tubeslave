# Brief: WING MVP Required Supervised Prestart Marker Count Summary Fields

## Task

Expose the top-level required supervised prestart marker count and count-match
verdicts in the no-WING handoff summary.

## Context

The no-WING handoff summary already exposes the structured supervised gate
marker count. It also compares that count to the top-level summary marker list,
but the top-level summary marker count itself was implicit. Operators and
Project Director should see both sides of that comparison directly.

## Scope

- Add top-level no-WING fields:
  `required_supervised_prestart_gate_marker_count`,
  `expected_required_supervised_prestart_gate_marker_count`,
  `required_supervised_prestart_gate_marker_count_matches_contract`, and
  `required_supervised_prestart_gate_marker_count_matches_supervised_gate`.
- Render them in JSON and Markdown summaries.
- Add Project Director baseline markers and tests.

## Out Of Scope

- Contacting WING hardware.
- Granting live authority.
- Closing readiness from offline evidence.
