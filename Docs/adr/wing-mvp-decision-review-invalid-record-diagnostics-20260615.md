# ADR: WING MVP Decision Review Invalid-Record Diagnostics

## 1. Context

The review-decision ledger rejects invalid operator records, including accepted
decision templates that still carry the generated pending `template_status`
marker. The standalone decision-review report exposed invalid ids, but not the
per-record violation list or stale template status in the operator-facing
markdown.

## 2. Options considered

- Leave invalid details only in the nested ledger payload.
- Add a special-case pending-template warning.
- Render a general invalid-record diagnostic list in the report.

## 3. Decision

Render a general `invalid_records` list in decision-review report JSON and
markdown.

## 4. Why this won

It keeps the ledger as the enforcement source while making the standalone
operator report self-contained. The same shape works for stale templates,
duplicate decisions, unsafe authority claims, and future invalid-record reasons.

## 5. Rejected alternatives

Nested-ledger-only diagnostics make hardware-session triage slower and easier
to misread. A special-case pending-template warning would leave other invalid
records hidden behind the same reporting gap.

## 6. Implementation plan

- Derive invalid-record rows from `ledger.records`.
- Add top-level report fields for review id, source id, decision,
  `template_status`, violations, and unsafe authority flags.
- Render those rows in markdown.
- Add a focused regression for `review_decision_template_status_unedited`.

## 7. Test plan

Run targeted decision-review tests, focused WING/no-WING tests, standard pytest,
no-WING quick, diff check, memory lint, and Project Director.

## 8. Risks and rollback

Risk is limited to report output. Rollback is removing the `invalid_records`
projection, markdown section, and focused regression.
