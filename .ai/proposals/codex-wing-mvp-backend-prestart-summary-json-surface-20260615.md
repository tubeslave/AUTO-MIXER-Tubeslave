# Proposal: WING MVP Backend Prestart Summary JSON Surface

Date: 2026-06-15

## 1. Short Thesis

Project the backend prestart summary binding already computed for the no-WING
handoff artifact into top-level `summary.json`.

## 2. What I Understood

The Current First Command requires a bound passed no-WING prestart
`summary.json` before any intended backend/WING session. Those fields are
available in the nested handoff artifact but not as compact top-level summary
fields for gate consumers.

## 3. Proposed Solution

Add a summary projection helper that copies the handoff artifact's backend
prestart summary path, SHA, type, status, policy, safety flags, and match
verdicts into top-level `summary.json`.

## 4. Likely Files to Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `Docs/INDEX.md`
- Council brief, proposal, ADR, and report files

## 5. Alternatives Considered

Keep the data nested only. That avoids duplication but leaves the pre-backend
gate hard to consume programmatically. Add another detached artifact. That is
unnecessary because the no-WING summary is already the release-gate surface.

## 6. Risks

The risk is making an offline prestart artifact look like live readiness. The
projection therefore explicitly keeps readiness closure and live authority
denied.

## 7. Test Plan

- Targeted no-WING summary test.
- no-WING quick.
- Project Director guard snapshot.
- Focused WING/no-WING/Director tests.
- Full standard pytest.
- Memory lint and `git diff --check`.

## 8. Where the Other Agent May Disagree

Kimi may prefer to avoid top-level duplication. The counterargument is that the
pre-backend gate is a release-gate contract, and top-level JSON is the easiest
stable surface for external automation to inspect.
