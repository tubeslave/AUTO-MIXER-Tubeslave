# ADR: WING MVP Accepted Candidate Expected Artifact Trace Summary JSON Surface

## 1. Context

The supervised WING MVP remains blocked on real read-only WING connection proof.
Before supervised proof planning, accepted review source ids must be bound to
the expected `summary.json` and `events.jsonl` artifacts. Existing no-WING
summary output exposed contract validity and violation maps, but not the trace
itself as a top-level summary surface.

## 2. Options considered

- Keep the trace nested inside candidate reviews only.
- Add top-level `expected_summary_artifact` fields unconditionally.
- Add a source-id keyed summary surface for accepted-candidate expected artifact
  traces.

## 3. Decision

Add a source-id keyed accepted-candidate expected artifact trace summary surface
to no-WING `summary.json`, and render the same fields in `summary.md`.

## 4. Why this won

It gives machine consumers and operators direct evidence without inventing a
real WING candidate. Empty no-WING runs remain explicit with empty maps and
`complete: false`.

## 5. Rejected alternatives

Unconditional top-level `expected_summary_artifact` was rejected because it
would blur the distinction between an expected glob and a reviewed real WING
artifact.

## 6. Implementation plan

- Compute trace fields from accepted source ids and `candidate_reviews`.
- Copy those fields into `next_wing_proof_plan`.
- Project them to top-level no-WING summary output.
- Extend the operator markdown required-field check.

## 7. Test plan

- Focused next-proof-plan summary test.
- no-WING quick run and `jq` field check.
- Project Director JSON check.
- Focused WING/no-WING suite.
- Full pytest, memory lint, and whitespace diff check.

## 8. Risks and rollback

Risk: downstream scripts may assume the old operator markdown required-field
list. Rollback is to remove the new required markdown fields and summary
surface while leaving existing candidate contract checks in place.
