# Brief: WING MVP Proof Plan Expected Prestart Violations Step Gate

## Task

Make no-WING validation fail closed when recomputed accepted-candidate prestart violations are non-empty while a supervised proof plan still contains physical proof steps.

## Context

The proof-plan summary carries both reported candidate prestart diagnostics and recomputed expected diagnostics. Existing validation compared stale summary maps against the decision-review ledger, but the recomputed expected map should directly block proof steps even if a stale summary tries to hide violations.

## Constraints

- No live mixer transport.
- No live-write authority.
- No fader or gain increase.
- Keep this scoped to no-WING validation.
- Preserve the first WING MVP blocker until real WING read-only evidence exists.

## Done When

- `next_wing_proof_plan_schema` rejects planned proof steps when recomputed expected accepted-candidate prestart violations are non-empty.
- The same condition requires `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof` to remain in `global_blockers`.
- A focused test covers the regression.
- ADR/report docs record the decision.
