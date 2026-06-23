# ADR: WING MVP Proof Plan Expected Prestart Violations Step Gate

## Context

The no-WING proof-plan validator recomputes expected accepted-candidate prestart diagnostics from the decision-review ledger. Before this decision, stale proof-plan summaries could fail through mismatch checks, but there was no dedicated failure saying that recomputed expected prestart violations forbid physical proof steps.

## Options considered

1. Keep relying on mismatch failures.
2. Add a direct fail-closed rule for non-empty recomputed expected prestart violations with planned proof steps.
3. Move the rule to docs only.

## Decision

Choose option 2.

`next_wing_proof_plan_schema` now reports `next_wing_proof_plan_steps_with_expected_candidate_prestart_violations` when recomputed expected accepted-candidate prestart violations are non-empty while proof steps are still planned.

## Why this won

The recomputed expected map is the validator's independent view of the decision-review ledger. If it is non-empty, physical fader/gain proof steps must remain blocked even when a stale reported proof-plan map is empty.

## Rejected alternatives

- Mismatch-only validation was indirect.
- Docs-only wording would not protect generated artifacts.

## Implementation plan

- Check recomputed `expected_candidate_prestart_violations_by_source` in `validate_supervised_proof_report_next_wing_plan()`.
- Fail when it is non-empty and `step_count` is non-zero.
- Require `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof` to remain in `global_blockers`.
- Add a regression test with an invalid accepted-candidate prestart summary status.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_steps_with_expected_candidate_prestart_violations`
- Existing focused no-WING/handoff validation.

## Risks and rollback

Risk: stale proof-plan artifacts now show an additional explicit violation. This is intended.

Rollback: remove the direct expected-map check and regression test. Mismatch checks remain, but the proof-step ban becomes indirect again.
