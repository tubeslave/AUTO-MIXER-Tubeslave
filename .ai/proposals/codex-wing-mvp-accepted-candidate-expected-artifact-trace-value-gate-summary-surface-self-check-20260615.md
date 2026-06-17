# Codex Proposal: WING MVP Accepted Candidate Expected Artifact Trace Value Gate Summary Surface Self-Check

## 1. Short thesis

Add a fail-closed self-check that proves the accepted-candidate expected
artifact trace value-gate fields match between top-level no-WING `summary.json`
and nested `next_wing_proof_plan`.

## 2. What I understood

The project still needs a real WING read-only connection proof before supervised
proof planning can advance. Current no-WING work should reduce handoff ambiguity
without contacting WING or granting authority.

## 3. Proposed solution

- Define the canonical value-gate summary-surface fields.
- Build a self-check with required fields, mismatch list, match flag, and
  readiness/live-authority denial fields.
- Fail the no-WING summary if any top-level value-gate field differs from the
  nested `next_wing_proof_plan` value.
- Render the mismatch list and match flag in `summary.md`.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## 5. Alternatives considered

- Rely on the current builder to keep fields in sync.
- Add markdown-only wording.
- Add a machine-readable self-check with fail-closed status.

## 6. Risks

The main risk is test/report churn around already-large no-WING summary
fixtures. The implementation should keep the comparison narrow to the six
value-gate fields.

## 7. Test plan

- Targeted no-WING tests for mismatch, summary embedding, operator markdown, and
  the new self-check failure path.
- no-WING quick run.
- Project Director check.
- Focused WING/no-WING suite.
- Full pytest, memory lint, and whitespace diff check.

## 8. Where the other agent may disagree

Kimi may argue this is another reporting rail instead of physical WING
progress. The counterpoint is that this rail protects the exact handoff artifact
operators use before the real WING read-only session and stays on the first
blocker.
