# Proposal: WING MVP Current Command Manifest Sidecar Checklist

## 1. Short Thesis

Mirror the runbook's no-WING `artifact-manifest.json` sidecar checklist into the
Project Director current first command so Goal-mode routing names the same
pre-WING gate operators must follow.

## 2. What I Understood

The read-only runbook now requires checking the detached manifest sidecar before
opening WING, but the Current First Command still omits those markers. Project
Director selects Current First Command as the active action, so the action text
must not lag behind the runbook.

## 3. Proposed Solution

Add the sidecar checklist markers to `.ai/project-layer/wing_mvp_blockers.md`
under `## Current First Command`, regenerate latest progress, and extend rails
tests to assert the command contains the required markers and denial language.

## 4. Likely Files to Touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tests/test_wing_mvp_rails.py`
- `Docs/INDEX.md`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`

## 5. Alternatives Considered

Leave the requirement only in the runbook. That preserves the procedure but
lets Project Director continue emitting an incomplete current command. Add a
new runtime gate. That is unnecessary because this is a directive alignment
gap, not a missing artifact field.

## 6. Risks

The Current First Command grows longer. The tradeoff is acceptable because this
command is already the full operator pre-WING checklist.

## 7. Test Plan

- Targeted rails test for current-command marker coverage.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Project Director guard snapshot.
- Focused WING/no-WING/Director tests.
- Memory lint and `git diff --check`.

## 8. Where the Other Agent May Disagree

Kimi may prefer a shorter action command that points to the runbook. The
counterpoint is that Project Director uses this text directly for Goal-mode
routing, so critical pre-WING safety gates should be explicit.
