# ADR: WING MVP Proof Plan Required Summary Field Gate

## Context

The WING handoff packet declares `required_supervised_proof_plan_summary_fields`. That contract now includes accepted-candidate prestart and artifact-root diagnostics. The no-WING proof-plan schema validator already compares specific diagnostics, but did not expose a single contract-level failure when a required field was absent from `next_wing_proof_plan.summary`.

## Options considered

1. Keep relying on specific mismatch checks.
2. Add a local hard-coded list of required proof-plan summary fields.
3. Reuse the shared handoff-required field tuple and fail closed on missing keys.

## Decision

Choose option 3.

`validate_supervised_proof_report_next_wing_plan` now reports:

- `required_supervised_proof_plan_summary_fields`
- `missing_required_supervised_proof_plan_summary_fields`

and emits `next_wing_proof_plan_required_summary_fields_missing` when the proof-plan summary omits any required field.

## Why this won

The handoff packet and no-WING proof-plan schema are two sides of the same operator contract. Using the shared tuple prevents drift and gives stale summaries a clear fail-closed reason before supervised proof planning.

## Rejected alternatives

- Specific mismatch checks alone are weaker because missing fields can collapse into empty defaults and make diagnosis less direct.
- A duplicated list would create another maintenance point and could drift from the handoff packet.

## Implementation plan

- Compute missing required fields from `WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS`.
- Add the expected and missing field lists to no-WING check detail.
- Add regression coverage for a missing accepted-candidate artifact-root field.
- Update WING blocker docs and report index.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_next_wing_proof_plan_schema tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_missing_required_summary_field`
- Run focused no-WING/handoff, rail, compile, memory-lint, and whitespace checks.

## Risks and rollback

Risk: stale generated proof-plan artifacts fail closed until regenerated with all required fields.

Rollback: remove the missing-field check and detail fields. This restores prior behavior but weakens the handoff-to-proof-plan contract.
