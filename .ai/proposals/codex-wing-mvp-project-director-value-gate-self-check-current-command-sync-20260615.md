# Codex Proposal: WING MVP Project Director Value Gate Self-Check Current Command Sync

## 1. Short thesis

Make Project Director's selected action carry the same value-gate self-check
requirements that no-WING summaries now enforce.

## 2. What I understood

Project Director is the audit/planning rail for continuous Goal-mode work. It
must direct work through the current first WING blocker and must not lag behind
the no-WING handoff contract.

## 3. Proposed solution

- Add value-gate self-check markers to `CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS`.
- Add the same markers to `.ai/project-layer/wing_mvp_blockers.md` under
  `## Current First Command`.
- Extend Project Director tests so selected actions must include the new
  self-check fields.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.

## 4. Likely files to touch

- `tools/codex_autoloop/update_mvp_progress.py`
- `tests/test_project_director_agent.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`
- `Docs/INDEX.md`

## 5. Alternatives considered

- Leave self-check markers only in the blocker table.
- Let Project Director keep choosing the older `Current First Command`.
- Require the self-check markers in the selected action baseline.

## 6. Risks

The marker tuple is large and can make unrelated current-command edits fail
baseline coverage. That is intentional for safety-critical operator handoff
requirements, but the markers should remain precise.

## 7. Test plan

- Targeted Project Director selected-action test.
- no-WING quick run.
- Project Director CLI check.
- Focused WING/no-WING suite.
- Full pytest, memory lint, and whitespace diff check.

## 8. Where the other agent may disagree

Kimi may prefer reducing the size of `Current First Command`. That can be a
future cleanup, but this patch keeps the current explicit contract and prevents
stale Project Director output.
