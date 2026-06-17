# ADR: WING MVP Proof Plan Prestart Violations Step Gate

## Context

The current WING MVP command says both accepted-candidate prestart violation maps must be `{}` before supervised proof steps are planned. Existing no-WING validation already compares proof-plan maps to the decision-review ledger, but a non-empty proof-plan prestart violation map did not have a dedicated "steps are forbidden" failure.

## Options considered

1. Keep relying on stale-map mismatch failures.
2. Add a direct no-WING validation rule for non-empty prestart violation maps with planned steps.
3. Move the check to docs only.

## Decision

Choose option 2.

`next_wing_proof_plan_schema` now reports `next_wing_proof_plan_steps_with_candidate_prestart_violations` when the proof-plan summary carries accepted-candidate prestart contract violations while proof steps are still planned.

## Why this won

The rule is a safety boundary, not just a consistency check. A non-empty accepted-candidate prestart violations map means the operator does not yet have a valid no-WING prestart trace for the accepted read-only candidate, so physical fader/gain proof steps must remain blocked.

## Rejected alternatives

- Stale-map mismatch alone was indirect.
- Docs-only wording would not protect generated artifacts.

## Implementation plan

- Check `readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source` in `validate_supervised_proof_report_next_wing_plan()`.
- Fail if the map is non-empty and `step_count` is non-zero.
- Require `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof` to remain in `global_blockers`.
- Add a regression test that keeps proof steps planned with a non-empty proof-plan violation map.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_steps_with_candidate_prestart_violations`
- Existing focused no-WING/handoff validation.

## Risks and rollback

Risk: stale summaries now produce an additional explicit violation. That is intended and should improve diagnosis.

Rollback: remove the direct check and regression test. The broader stale-map checks remain, but the operator-facing "no steps with violations" rule becomes indirect again.
