# ADR: WING MVP Accepted Candidate Expected Artifact Trace Required Fields

## 1. Context

The WING supervised proof path requires a reviewed read-only preflight
candidate before any physical proof. The accepted-candidate expected artifact
trace must stay visible and machine-checkable so `expected_artifacts`,
`expected_summary_artifact`, `expected_events_artifact`, and match flags cannot
silently disappear from the proof-plan summary.

## 2. Options considered

- Leave the trace as a top-level no-WING surface only.
- Add markdown-only operator checklist text.
- Add trace fields to the canonical required proof-plan summary field contract.

## 3. Decision

Add the accepted-candidate expected artifact trace fields to
`WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS`
and have `build_supervised_wing_proof_plan()` write them in
`next_wing_proof_plan.summary`.

## 4. Why this won

The required-field contract is already enforced by no-WING validation and by
the handoff packet checks. Extending it makes missing trace fields a schema
failure instead of a weak report-only gap.

## 5. Rejected alternatives

Top-level summary-only evidence was rejected because it would not prove the
generated proof plan itself carries the trace. Markdown-only evidence was
rejected because it is less reliable for automated gates.

## 6. Implementation plan

- Extend the shared required-field tuple.
- Add a small trace helper in the supervised proof-plan builder.
- Insert the trace fields into the proof-plan summary.
- Update tests and fixtures that assert the required-field contract.

## 7. Test plan

- Focused supervised proof-plan tests.
- Focused no-WING required-field and summary tests.
- no-WING quick artifact check.
- Project Director check.
- Focused WING/no-WING suite.
- Full pytest, memory lint, and whitespace diff check.

## 8. Risks and rollback

Risk: downstream fixtures that use the shared required-field tuple must include
the new fields. Rollback is to remove the trace fields from the tuple and
builder summary while keeping the previously added top-level summary surface.
