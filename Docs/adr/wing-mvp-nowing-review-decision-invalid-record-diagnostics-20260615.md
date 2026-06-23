# ADR: WING MVP no-WING Review-Decision Invalid-Record Diagnostics

## 1. Context

The WING read-only review-decision ledger fails closed when operator decision
records are invalid. The standalone decision-review report now exposes invalid
record details, but the no-WING release summary still showed only invalid
counts and ids.

## 2. Options considered

- Leave no-WING summaries with ids only.
- Recompute invalid decision violations in no-WING.
- Mirror invalid ledger records into no-WING summary JSON and markdown.

## 3. Decision

Mirror invalid ledger record diagnostics in no-WING `summary.json` and
`summary.md`.

## 4. Why this won

It keeps the review-decision ledger as the validation source while making the
quick release gate operator-readable. Operators can see stale pending
`template_status` markers and exact violations without digging through nested
records.

## 5. Rejected alternatives

Ids-only output was rejected because it hides the actionable cause. Recomputing
violations in no-WING was rejected because it would duplicate ledger policy.

## 6. Implementation plan

- Normalize invalid records from ledger `records`.
- Add `invalid_records` to no-WING schema detail.
- Propagate it through the no-WING summary builder.
- Render it in markdown.
- Add focused validator and summary tests.

## 7. Test plan

Run targeted no-WING tests, focused WING/no-WING tests, standard pytest,
no-WING quick, diff check, memory lint, and Project Director.

## 8. Risks and rollback

Risk is limited to report output. Rollback is removing the invalid-record
projection from no-WING detail, summary JSON, markdown, and tests.
