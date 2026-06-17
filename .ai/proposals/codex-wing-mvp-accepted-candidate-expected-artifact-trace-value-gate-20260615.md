# Proposal: WING MVP Accepted Candidate Expected Artifact Trace Value Gate

## 1. Short thesis

Required trace fields must also be value-gated against the review decision
ledger, not only present in the proof-plan summary.

## 2. What I understood

The current first WING command depends on accepted source ids being bound to
expected read-only preflight artifacts before any supervised proof step. The
previous patch promoted the trace fields into the required summary contract.
This patch closes the remaining gap: stale or falsified trace values should
fail no-WING validation.

## 3. Proposed solution

- Reuse the existing canonical accepted-candidate expected artifact trace
  helper in `tools/no_wing_release_check.py`.
- Add small exact-map normalizers for string-list maps, string-value maps, and
  bool maps.
- Compare every reported `next_wing_proof_plan.summary` trace field against
  the canonical trace computed from `wing_readonly_preflight_review_decision_ledger.summary.candidate_reviews`.
- Add a parametrized regression test that mutates each trace field and expects
  a specific mismatch violation.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- Council brief/proposal/ADR/report and docs index/blocker references

## 5. Alternatives considered

- Rely on required-field presence only. Rejected because a present field can
  still be stale or incorrect.
- Compare only the completion boolean. Rejected because it would not prove the
  source-id keyed artifact maps are correct.

## 6. Risks

The check is stricter and may expose fixtures or generated reports that copied
old trace values. That is the intended fail-closed behavior; the rollback is to
remove the value comparisons while keeping the required-field presence gate.

## 7. Test plan

- Targeted no-WING trace mismatch regression test.
- Existing next proof-plan summary surface test.
- no-WING quick run and summary field check.
- Project Director check.
- Focused WING/no-WING suite.
- Full pytest, memory lint, and `git diff --check`.

## 8. Where the other agent may disagree

Kimi may ask to move the trace value gate into the shared backend contract
module. Keeping this patch inside no-WING validation is lower risk because the
canonical proof-plan builder already emits the values and the release gate is
where stale report artifacts are rejected.
