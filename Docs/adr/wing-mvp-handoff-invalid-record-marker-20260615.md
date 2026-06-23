# ADR: WING MVP Handoff Invalid-Record Marker

## 1. Context

The supervised proof-plan summary contract includes
`readonly_preflight_review_decision_invalid_records`, and no-WING validates the
proof-plan field against the review-decision ledger. The WING read-only handoff
packet carried the full required-field list but did not expose a direct marker
that the invalid-record field was included.

## 2. Options considered

- Leave the field visible only inside the required-field list.
- Add documentation only.
- Add an explicit handoff/no-WING marker and fail closed when it is stale.

## 3. Decision

Add `required_supervised_proof_plan_summary_fields_include_invalid_records` to
the handoff packet, handoff markdown, no-WING handoff artifact validation,
no-WING summary output, Current First Command, and operator docs.

## 4. Why this won

It gives operators and release checks a direct proof that invalid review-decision
records remain part of the supervised proof-plan boundary before any physical
proof can be considered.

## 5. Rejected alternatives

The list-only approach was rejected because it is harder to inspect and easier
to miss in long handoff output. Documentation-only was rejected because stale
packet JSON would still pass.

## 6. Implementation plan

- Compute the marker from the required proof-plan summary field list.
- Render it in handoff markdown.
- Validate the declared marker in no-WING.
- Propagate it into no-WING summary JSON and markdown.
- Add focused regression tests and Council docs.

## 7. Test plan

Run handoff packet tests, no-WING handoff tests, Project Director tests,
WING-rails tests, no-WING quick, Project Director JSON, diff check, memory
lint, and standard pytest.

## 8. Risks and rollback

Risk is limited to offline reporting and validation. Rollback is removing the
marker from packet generation, no-WING validation, summaries, docs, and tests.
