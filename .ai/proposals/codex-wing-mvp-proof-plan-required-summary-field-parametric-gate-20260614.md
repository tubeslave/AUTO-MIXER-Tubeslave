# Proposal: WING MVP proof-plan required summary field parametric gate

## 1. Short thesis

The no-WING release tests should prove the required summary field gate for every
canonical supervised proof-plan field.

## 2. What I understood

The selected WING rail requires `next_wing_proof_plan_schema` to fail closed with
`next_wing_proof_plan_required_summary_fields_missing` when any required summary
field is absent. Existing coverage checked one field only.

## 3. Proposed solution

Parameterize the existing missing-required-field regression over
`WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS`.

## 4. Likely files to touch

- `tests/test_no_wing_release_check.py`
- Council brief, ADR, and report

## 5. Alternatives considered

Leaving one representative test would keep the suite smaller, but it would not
prove the field-by-field contract now shown in the Project Director directive.

## 6. Risks

The change increases test count by a small amount only. It does not alter runtime
or mixer behavior.

## 7. Test plan

Run the parametrized test and focused WING rail suites.

## 8. Where the other agent may disagree

Kimi may consider this redundant because the validator loop is generic. The
counterpoint is that the user-facing WING gate claims every required field, so
the regression should match that claim.
