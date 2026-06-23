# ADR: WING MVP Proof Plan Artifact Root Trace

## Context

Accepted WING read-only preflight candidates now carry `artifact_root` through capture, evidence intake, manual review decisions, and decision-review reports. The supervised proof plan already exposes accepted source ids and prestart diagnostics, but did not independently show the accepted-candidate root trace.

## Options considered

1. Leave proof-plan summaries unchanged and rely on decision-review reports.
2. Add artifact-root fields to proof-plan summaries without validation.
3. Add artifact-root fields and fail closed when accepted candidates lack a valid root trace.

## Decision

Choose option 3.

The supervised proof-plan summary now exposes:

- `readonly_preflight_review_decision_candidate_artifact_root_contract_valid`
- `readonly_preflight_review_decision_candidate_artifact_roots`
- `readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source`
- `readonly_preflight_review_decision_expected_candidate_artifact_roots`
- `readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source`

## Why this won

The operator-facing proof plan is the next downstream surface before physical supervised proof. It should be able to prove that the accepted read-only candidate still points at the same artifact root before any fader/gain proof step is planned.

## Rejected alternatives

- Reusing the prestart blocker would hide whether a failure came from no-WING prestart metadata or missing artifact-root trace.
- Rendering fields without validation would create a display-only trace that stale summaries could bypass.

## Implementation plan

- Add a shared blocker constant for accepted-candidate artifact-root failures.
- Add local proof-plan/no-WING helpers to derive root diagnostics from accepted source ids and `candidate_reviews`.
- Set `readonly_preflight_review_decision_accepted` false when the root contract is invalid.
- Add focused regression coverage for missing root and stale no-WING summaries.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_supervised_wing_proof_plan.py`
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py`
- Run rail, compile, memory-lint, and whitespace checks before handoff.

## Risks and rollback

Risk: older generated proof-plan summaries fail closed until regenerated.

Rollback: remove the artifact-root proof-plan fields and blocker check. This would restore prior behavior but would weaken the downstream trace from accepted read-only candidate to supervised proof plan.
