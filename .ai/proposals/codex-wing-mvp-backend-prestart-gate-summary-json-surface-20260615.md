# Proposal: WING MVP Backend Prestart Gate Summary JSON Surface

Date: 2026-06-15

## 1. Short Thesis

Project the backend prestart gate verdicts already computed for the no-WING
handoff artifact into top-level `summary.json`.

## 2. What I Understood

The Current First Command requires the prestart gate to match its expected
contract and requires the command text to be visible before any backend/WING
session. The values are present in the nested handoff artifact and rendered in
Markdown, but top-level `summary.json` consumers currently have to parse the
nested packet to inspect them.

## 3. Proposed Solution

Add a small summary projection helper that copies the validated handoff values
into top-level `summary.json` and adds explicit `false` readiness/live-authority
flags for the projection.

## 4. Likely Files to Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `Docs/INDEX.md`
- Council brief, proposal, ADR, and report files

## 5. Alternatives Considered

Keep the fields nested only. That keeps code smaller but forces gate consumers
to parse a large packet blob. Add a separate artifact. That adds a file contract
without a clear benefit over the existing summary.

## 6. Risks

The main risk is confusing offline reporting with live authority. The projection
therefore carries only existing verdicts and explicitly denies readiness closure
and live authority.

## 7. Test Plan

- Targeted no-WING summary test.
- no-WING quick.
- Project Director guard snapshot.
- Focused WING/no-WING/Director tests.
- Full standard pytest.
- Memory lint and `git diff --check`.

## 8. Where the Other Agent May Disagree

Kimi may argue that duplicating fields in top-level JSON adds contract sprawl.
The counterargument is that this is a release-gate machine surface, and the
duplicated values are already stable handoff verdicts.
