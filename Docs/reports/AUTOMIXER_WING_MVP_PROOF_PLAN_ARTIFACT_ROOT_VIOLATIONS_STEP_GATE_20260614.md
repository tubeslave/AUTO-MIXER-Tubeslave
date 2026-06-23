# WING MVP Proof Plan Artifact-Root Violations Step Gate

## Summary

The no-WING release checker now has an explicit fail-closed gate for supervised proof plans that still carry accepted-candidate artifact-root contract violations.

If `next_wing_proof_plan.summary.readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source` is non-empty while physical proof steps are planned, `next_wing_proof_plan_schema` fails with:

- `next_wing_proof_plan_steps_with_candidate_artifact_root_violations`

The same condition also requires the existing blocker:

- `accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof`

## Why It Matters

The current WING MVP rail requires accepted read-only evidence to remain bound to its expected artifact root before any supervised fader/gain proof. A non-empty artifact-root violations map means proof planning cannot be trusted to point at the intended candidate.

## Safety Boundary

- No live WING access was added.
- No mixer state mutation was added.
- No fader or gain increase was added.
- The first open blocker remains `WING read-only connection proof`.

## Validation

Added a regression test:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_steps_with_candidate_artifact_root_violations
```

Focused no-WING/handoff validation should continue to cover the broader summary contract.
