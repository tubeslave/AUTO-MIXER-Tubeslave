# Brief: WING MVP Supervised Prestart Marker Count Summary Fields

## Task

Expose the supervised prestart gate required marker count and count-match
verdicts in the no-WING handoff summary.

## Context

The WING supervised proof rail already exposes the required marker list and
marker-list verdicts. Operators and Project Director should also see the
expected count and count-match verdicts directly, so missing or extra marker
shape changes are visible without parsing nested gate objects.

## Scope

- Add top-level no-WING fields:
  `supervised_prestart_gate_required_marker_count`,
  `expected_supervised_prestart_gate_required_marker_count`,
  `supervised_prestart_gate_required_marker_count_matches_contract`, and
  `supervised_prestart_gate_required_marker_count_matches_summary`.
- Render them in JSON and Markdown summaries.
- Add Project Director baseline markers and tests.

## Out Of Scope

- Contacting WING hardware.
- Granting live authority.
- Closing readiness from offline evidence.
