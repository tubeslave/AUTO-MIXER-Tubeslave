# WING MVP proof-plan expected artifact-root violations step gate

Date: 2026-06-14

## Summary

The no-WING release validator now rejects physical WING proof steps when the
recomputed
`readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source`
map is non-empty.

## Safety impact

- Grants no live authority.
- Keeps the current first blocker on WING read-only connection proof.
- Preserves the accepted-candidate artifact-root blocker until the real
  read-only evidence binds to the expected artifact root.

## Validation target

The regression removes the accepted candidate `artifact_root` from the review
ledger and expects
`next_wing_proof_plan_steps_with_expected_candidate_artifact_root_violations`
while proof steps are still present.
