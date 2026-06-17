# Proposal: WING MVP Project Director Value Gate Self-Check Markdown Surface

## 1. Short thesis

Add a runtime no-WING cross-surface check that proves Project Director
value-gate self-check fields are visible in the operator Markdown handoff.

## 2. What I understood

The first WING MVP blocker still waits for a real read-only WING artifact. The
offline handoff should keep making that hardware step safer and clearer, but it
must not approve writes, close readiness, or grant live authority.

## 3. Proposed solution

- Define required Markdown field lines for the Project Director selected-action
  value-gate self-check surface.
- Add an applicator that scans `summary.md`, records missing lines, and fails
  no-WING validation if any are absent.
- Render the Markdown-surface verdict into both `summary.json` and `summary.md`.
- Require these fields in the current-first Project Director baseline.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `tests/test_project_director_agent.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`

## 5. Alternatives considered

- Trust tests only.
- Trust `summary.json` only.
- Add prose instructions without a failing runtime check.

## 6. Risks

The no-WING summary becomes stricter: if Markdown rendering drifts, the release
check fails even when JSON is correct. That is appropriate because the operator
handoff relies on Markdown visibility before real hardware work.

## 7. Test plan

- Unit test for missing Project Director Markdown lines.
- Summary integration assertions for JSON and Markdown fields.
- Project Director selected-action baseline test.
- no-WING quick, focused WING/no-WING tests, full pytest, memory lint, and diff
  check.

## 8. Where the other agent may disagree

Kimi may argue the JSON surface is sufficient. The counterpoint is the runbook
is operator-facing: a real preflight can start from Markdown review, so Markdown
visibility must be a release gate too.
