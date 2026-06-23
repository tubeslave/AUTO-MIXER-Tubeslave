# WING MVP proof-plan required summary field parametric gate

Date: 2026-06-14

## Summary

The no-WING release regression for missing supervised proof-plan summary fields
now runs against every canonical required field in
`WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS`.

## Safety impact

- Grants no live authority.
- Changes no mixer runtime behavior.
- Strengthens evidence that incomplete proof-plan summaries fail closed before
  supervised WING proof steps proceed.

## Validation target

For each required summary field, remove that field from a valid proof-plan
summary and require `next_wing_proof_plan_required_summary_fields_missing`.
