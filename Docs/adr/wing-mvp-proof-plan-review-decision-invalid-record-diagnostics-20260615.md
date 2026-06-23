# ADR: WING MVP Proof-Plan Review-Decision Invalid-Record Diagnostics

## 1. Context

The WING read-only review-decision ledger fails closed when operator decision
records are invalid. Standalone decision-review reports and no-WING
review-decision summaries now expose invalid record `template_status` and
`violations`, but the supervised proof-plan summary still exposed only counts
and ids.

## 2. Options considered

- Leave proof-plan summaries with invalid counts and ids only.
- Recompute invalid review-decision violations in proof-plan/no-WING logic.
- Project normalized invalid ledger records into the proof-plan summary and
  validate no-WING consistency against the ledger.

## 3. Decision

Project normalized invalid review-decision records into
`next_wing_proof_plan.summary`, make the field part of the required proof-plan
summary contract, and have no-WING validation fail closed on mismatch.

## 4. Why this won

It keeps the ledger as the validation source while making the next operator
proof-plan surface self-explanatory. A stale pending review template now remains
visible in `summary.json` and `summary.md` wherever the proof plan is consumed.

## 5. Rejected alternatives

Counts and ids only were rejected because they hide the actionable reason.
Recomputing the validation in no-WING was rejected because it would duplicate
ledger policy and increase drift risk.

## 6. Implementation plan

- Normalize invalid records in the supervised proof-plan builder.
- Add the field to the proof-plan summary contract.
- Validate proof-plan invalid records against ledger invalid records in
  no-WING.
- Propagate the field through no-WING JSON and markdown summaries.
- Add targeted regression tests.

## 7. Test plan

Run targeted proof-plan/no-WING tests, focused WING/no-WING tests, standard
pytest, no-WING quick, diff check, memory lint, and Project Director.

## 8. Risks and rollback

Risk is limited to offline reporting and schema fixtures. Rollback is removing
the proof-plan invalid-record field, no-WING mismatch check, markdown rendering,
and tests.
