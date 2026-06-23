# ADR: WING MVP Proof Plan Artifact-Root Violations Step Gate

## Context

The current WING MVP command requires accepted read-only candidates to carry a valid accepted-candidate `artifact_root` trace before supervised fader/gain proof steps are planned. Existing no-WING validation already compared proof-plan artifact-root diagnostics to the decision-review ledger, but a non-empty proof-plan artifact-root violation map did not have a dedicated "steps are forbidden" failure.

## Options considered

1. Keep relying on stale-map mismatch failures.
2. Add a direct no-WING validation rule for non-empty artifact-root violation maps with planned steps.
3. Move the check to docs only.

## Decision

Choose option 2.

`next_wing_proof_plan_schema` now reports `next_wing_proof_plan_steps_with_candidate_artifact_root_violations` when the proof-plan summary carries accepted-candidate artifact-root contract violations while proof steps are still planned.

## Why this won

The rule is a safety boundary, not just a consistency check. A non-empty accepted-candidate artifact-root violations map means the operator cannot reliably bind the accepted read-only evidence to the future supervised proof plan, so physical fader/gain proof steps must remain blocked.

## Rejected alternatives

- Stale-map mismatch alone was indirect.
- Docs-only wording would not protect generated artifacts.

## Implementation plan

- Check `readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source` in `validate_supervised_proof_report_next_wing_plan()`.
- Fail if the map is non-empty and `step_count` is non-zero.
- Require `accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof` to remain in `global_blockers`.
- Add a regression test that keeps proof steps planned with a non-empty proof-plan artifact-root violation map.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_steps_with_candidate_artifact_root_violations`
- Existing focused no-WING/handoff validation.

## Risks and rollback

Risk: stale summaries now produce an additional explicit violation. That is intended and should improve diagnosis.

Rollback: remove the direct check and regression test. The broader stale-map checks remain, but the operator-facing "no steps with artifact-root violations" rule becomes indirect again.
