# Brief: WING MVP Proof Plan Artifact-Root Violations Step Gate

## Task

Make no-WING validation fail closed when a supervised proof plan still carries accepted-candidate artifact-root contract violations while also planning physical proof steps.

## Context

The current WING MVP command requires accepted read-only candidates to have both a valid no-WING prestart trace and a valid accepted-candidate `artifact_root` trace before supervised fader/gain proof can be planned. Existing validation already compared artifact-root diagnostics against the decision-review ledger, but the "non-empty artifact-root map means no proof steps" rule should be direct and diagnosable.

## Constraints

- No live mixer transport.
- No live-write authority.
- No fader or gain increase.
- Keep this scoped to the no-WING validation gate.
- Preserve the first WING MVP blocker until real WING read-only evidence exists.

## Done When

- `next_wing_proof_plan_schema` rejects planned proof steps when `readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source` is non-empty.
- The same condition requires `accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof` to remain in `global_blockers`.
- A focused test covers the regression.
- ADR/report docs record the decision.
