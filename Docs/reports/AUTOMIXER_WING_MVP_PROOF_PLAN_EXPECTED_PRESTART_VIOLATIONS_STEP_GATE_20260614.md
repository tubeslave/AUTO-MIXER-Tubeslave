# WING MVP Proof Plan Expected Prestart Violations Step Gate

## Summary

The no-WING release checker now directly blocks supervised proof steps when recomputed expected accepted-candidate prestart violations are non-empty.

If the decision-review ledger implies accepted-candidate prestart violations while `next_wing_proof_plan` still plans physical proof steps, `next_wing_proof_plan_schema` fails with:

- `next_wing_proof_plan_steps_with_expected_candidate_prestart_violations`

The same condition also requires:

- `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`

## Why It Matters

The expected map is recomputed from the ledger, so it is the validator's independent check against stale or over-optimistic proof-plan summaries. Physical fader/gain proof must not proceed when the recomputed view still finds an invalid no-WING prestart trace.

## Safety Boundary

- No live WING access was added.
- No mixer state mutation was added.
- No fader or gain increase was added.
- The first open blocker remains `WING read-only connection proof`.

## Validation

Added a regression test:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_steps_with_expected_candidate_prestart_violations
```

Focused no-WING/handoff validation should continue to cover the broader summary contract.
