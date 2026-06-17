# Proposal: WING MVP Accepted Candidate Expected Artifact Trace Required Fields

## 1. Short thesis

Promote the accepted-candidate expected artifact trace from an exposed surface
to a required proof-plan summary contract.

## 2. What I understood

The current first WING command requires accepted source ids to have a valid
expected-artifact-bound trace before supervised proof. A summary surface helps
inspection, but the schema should also fail closed if the proof plan omits the
trace fields.

## 3. Proposed solution

- Extend the canonical required proof-plan summary field tuple.
- Compute accepted-candidate expected artifact trace inside
  `build_supervised_wing_proof_plan()`.
- Add the trace fields to `next_wing_proof_plan.summary`.
- Update focused tests and fixtures so missing-field validation covers the new
  fields.

## 4. Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/supervised_wing_proof_plan.py`
- `tests/test_no_wing_release_check.py`
- Council brief/proposal/ADR/report and docs index/blocker references

## 5. Alternatives considered

- Keep the trace as top-level no-WING summary-only evidence. Rejected because a
  generated proof plan could still omit the fields without the required-field
  contract noticing.
- Add only a markdown checklist. Rejected because machine validation is the
  safer boundary.

## 6. Risks

The required-field tuple is shared by handoff packet and no-WING validation, so
fixtures and generated handoff artifacts must remain synchronized.

## 7. Test plan

- Supervised WING proof-plan tests.
- no-WING required-field and summary surface tests.
- no-WING quick run and `jq` field check.
- Project Director check.
- Focused WING/no-WING suite.
- Full pytest, memory lint, and `git diff --check`.

## 8. Where the other agent may disagree

Kimi may ask for value-mismatch checks as a separate hard gate. This patch keeps
the scope to required-field presence because the current accepted-candidate
contract maps already gate physical proof planning, and the trace values are
computed from candidate reviews.
