# Brief: WING MVP proof-plan required summary field parametric gate

## Task

Strengthen the no-WING regression coverage for the supervised proof-plan
required summary field contract.

## Scope

- Keep the active MVP on supervised WING Rack proof.
- Do not change live authority or mixer behavior.
- Prove that `next_wing_proof_plan_required_summary_fields_missing` applies to
  every canonical supervised prestart proof-plan summary field, not just one
  representative field.

## Test Plan

- Run the parametrized missing-required-field regression.
- Run focused no-WING/handoff and Project Director suites.
