# Brief: WING MVP Proof Plan Prestart Violations Step Gate

## Task

Make no-WING validation fail closed when a supervised proof plan still carries accepted-candidate prestart contract violations while also planning physical proof steps.

## Context

The current WING MVP command requires both the decision-review ledger/report map and the supervised proof-plan summary map to be `{}` before any supervised proof steps are planned. Existing validation already compares stale maps against the decision-review ledger, but the "non-empty map means no proof steps" rule should be direct and diagnosable.

## Constraints

- No live mixer transport.
- No live-write authority.
- No fader or gain increase.
- Keep this scoped to the no-WING validation gate.
- Preserve the first WING MVP blocker until real WING read-only evidence exists.

## Done When

- `next_wing_proof_plan_schema` rejects planned proof steps when `readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source` is non-empty.
- The same condition requires `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof` to remain in `global_blockers`.
- A focused test covers the regression.
- ADR/report docs record the decision.
