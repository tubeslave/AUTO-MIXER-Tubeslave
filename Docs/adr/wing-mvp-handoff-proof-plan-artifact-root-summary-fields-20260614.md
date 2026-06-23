# ADR: WING MVP Handoff Proof Plan Artifact Root Summary Fields

## Context

Accepted-candidate `artifact_root` diagnostics are now present in supervised proof-plan summaries and no-WING validation. The WING read-only handoff packet already declares which proof-plan summary fields must exist before supervised proof planning, but the artifact-root diagnostics were not part of that required field contract.

## Options considered

1. Leave the handoff field contract unchanged and rely on proof-plan validation.
2. Add only expected artifact-root fields to the handoff field contract.
3. Add the full accepted-candidate artifact-root diagnostic set to the shared required proof-plan summary field contract.

## Decision

Choose option 3.

`WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS` now includes:

- `readonly_preflight_review_decision_candidate_artifact_root_contract_valid`
- `readonly_preflight_review_decision_candidate_artifact_roots`
- `readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source`
- `readonly_preflight_review_decision_expected_candidate_artifact_roots`
- `readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source`

## Why this won

The handoff packet is the operator-facing contract before physical supervised proof planning. It should require the same accepted-candidate root diagnostics that the proof-plan/no-WING layer validates, so a stale handoff cannot omit the root trace while still appearing contract-complete.

## Rejected alternatives

- Relying only on proof-plan validation would make the handoff summary weaker than the downstream gate.
- Adding only expected fields would miss the actual field names that operators and validators use to compare the computed root contract.

## Implementation plan

- Extend the shared required proof-plan summary field tuple in the contract module.
- Assert the root fields in handoff packet and no-WING validation tests.
- Assert the current command and Project Director selected action include the root fields.
- Update WING blocker docs, report docs, and the documentation index.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py`
- `PYTHONPATH=backend python -m pytest -q tests/test_wing_mvp_rails.py tests/test_project_director_agent.py`
- Compile changed modules and run rail/lint checks.

## Risks and rollback

Risk: older handoff/no-WING artifacts that do not declare the artifact-root fields will fail closed until regenerated.

Rollback: remove the artifact-root fields from the shared required tuple and related assertions. This restores the prior handoff contract but weakens traceability from accepted read-only evidence into supervised proof planning.
