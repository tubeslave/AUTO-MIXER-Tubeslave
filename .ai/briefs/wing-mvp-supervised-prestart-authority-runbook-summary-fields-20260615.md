# Brief: WING MVP Supervised Prestart Authority Runbook Summary Fields

## Task

Expose supervised prestart authority and runbook gates as top-level no-WING
summary fields.

## Context

The structured `supervised_prestart_gate` already states that it does not grant
live authority, requires `accepted_decision_ready`, and unlocks
`Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`. These safety fields should
be visible without parsing the nested gate object.

## Scope

- Add top-level no-WING fields for:
  `supervised_prestart_gate_grants_live_authority`,
  `supervised_prestart_gate_requires_operator_decision_status`, and
  `supervised_prestart_gate_unlocks_runbook`.
- Mirror the fields in JSON and Markdown summaries.
- Add Project Director baseline markers.

## Out Of Scope

- Granting live authority.
- Running WING hardware.
- Closing the WING read-only connection proof blocker.
