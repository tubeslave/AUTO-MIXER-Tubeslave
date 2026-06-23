# Proposal: WING MVP Command Binding Summary JSON Surface

Date: 2026-06-15

## 1. Short Thesis

Project the WING read-only command binding verdicts already computed in the
handoff artifact into top-level no-WING `summary.json`.

## 2. What I Understood

The Current First Command requires preview, capture, review, and decision-review
commands to bind to the expected prestart summary, artifact roots, plan output,
and decision roots before any intended WING session. These fields are currently
available inside the handoff artifact and operator Markdown, but not as compact
top-level JSON fields.

## 3. Proposed Solution

Add a summary projection helper that copies the existing handoff command
binding values into top-level `summary.json` and keeps authority denied.

## 4. Likely Files to Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `Docs/INDEX.md`
- Council brief, proposal, ADR, and report files

## 5. Alternatives Considered

Keep the data nested only. That avoids duplication but makes external gate
automation parse a large handoff artifact. Add a new artifact. That adds file
surface without changing the underlying source of truth.

## 6. Risks

The risk is presenting command binding readiness as live readiness. The
projection therefore carries only offline verdicts and explicitly denies
readiness closure and live authority.

## 7. Test Plan

- Targeted no-WING summary test.
- no-WING quick.
- Project Director guard snapshot.
- Focused WING/no-WING/Director tests.
- Full standard pytest.
- Memory lint and `git diff --check`.

## 8. Where the Other Agent May Disagree

Kimi may argue that top-level duplication increases the summary contract size.
The counterargument is that command binding is a pre-live gate and deserves a
stable machine-readable surface.
