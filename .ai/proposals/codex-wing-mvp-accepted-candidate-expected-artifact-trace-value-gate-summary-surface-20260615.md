# Proposal: WING MVP Accepted Candidate Expected Artifact Trace Value Gate Summary Surface

## 1. Short thesis

Make the accepted-candidate expected artifact trace value gate visible as a
first-class no-WING summary surface.

## 2. What I understood

The validator already rejects stale trace values, but operators need the
specific mismatch list in `summary.json` and `summary.md`, not only nested in
the raw check-result internals.

## 3. Proposed solution

- Add a canonical tuple of accepted-candidate expected artifact trace value-gate
  violation ids.
- Add value-gate required, violation-id, current-violation, passed, and
  safety-denial fields to `next_wing_proof_plan_schema.detail`.
- Mirror those fields through `next_wing_proof_plan_summary()` and a top-level
  `build_next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface()`.
- Render value-gate fields in the operator `summary.md`.
- Extend tests so mismatched trace fields surface their exact violation and a
  valid report shows `passed: true` with no violations.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- Council brief/proposal/ADR/report and docs index/blocker references

## 5. Alternatives considered

- Leave the mismatch list only inside raw check-result `violations`. Rejected
  because it makes operator triage and cross-surface checks weaker.
- Add only markdown text. Rejected because no-WING automation needs JSON fields.

## 6. Risks

The surface adds more required operator markdown lines. The risk is stale test
fixtures or generated summaries missing the new lines; that is acceptable
because the output should fail closed until regenerated.

## 7. Test plan

- Targeted no-WING trace mismatch and summary embedding tests.
- no-WING quick run.
- Project Director check.
- Focused WING/no-WING suite.
- Full pytest, memory lint, and `git diff --check`.

## 8. Where the other agent may disagree

Kimi may ask to consolidate this with the previous value-gate patch. Keeping it
separate documents the difference between fail-closed validation and
operator-visible reporting.
