# Proposal: WING MVP Accepted Candidate Expected Artifact Trace Summary JSON Surface

## 1. Short thesis

Add a no-WING summary surface that makes accepted-candidate expected artifact
trace fields explicit without changing readiness or granting live authority.

## 2. What I understood

The Project Director first command now depends on accepted review source ids
being bound to exact WING preflight artifacts before supervised proof. The
proof plan summary should make that trace directly inspectable.

## 3. Proposed solution

- Compute accepted-candidate expected artifact trace from accepted source ids
  and `candidate_reviews`.
- Copy the trace into `next_wing_proof_plan`.
- Project the trace to top-level no-WING `summary.json`.
- Render the same fields in the operator markdown proof-plan section.
- Keep the current no-candidate state explicit: empty maps and `complete: false`.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/INDEX.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- this brief/proposal/ADR/report set

## 5. Alternatives considered

- Add only documentation: rejected because machine consumers would still need to
  infer the trace from nested or external artifacts.
- Populate `expected_summary_artifact` top-level directly: rejected because that
  would imply a concrete accepted WING run when none exists.

## 6. Risks

The main risk is accidentally making an empty no-WING state look accepted. The
surface avoids that by using source-id maps and `complete: false` when no
accepted candidate exists.

## 7. Test plan

- Focused next-proof-plan summary test.
- no-WING quick run with `--skip-frontend`.
- Project Director JSON check.
- Focused WING/no-WING suite.
- Full pytest.
- memory lint and `git diff --check`.

## 8. Where the other agent may disagree

Kimi may prefer adding a hard validation failure for missing trace fields. I
would keep this increment as a surface first because existing candidate
contract checks already fail closed, and the current no-WING state legitimately
has no accepted candidate.
