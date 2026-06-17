# Proposal: WING MVP Project Director Value Gate Self-Check Summary Surface

## 1. Short thesis

Add a no-WING summary surface that explicitly proves Project Director's
selected action still contains the value-gate self-check markers required
before any supervised proof planning.

## 2. What I understood

The current MVP blocker remains `WING read-only connection proof`. Offline
validation can harden operator handoff contracts, but it cannot contact WING,
approve writes, close readiness, or grant live authority.

## 3. Proposed solution

- Add a canonical marker tuple for Project Director selected-action value-gate
  self-check requirements.
- Validate the handoff selected action against that marker tuple.
- Publish required, missing, present, readiness-denial, and live-authority
  denial fields in `summary.json` and `summary.md`.
- Add the same field names to the current-first baseline so Project Director
  keeps directing operators to inspect them.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_project_director_agent.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`

## 5. Alternatives considered

- Rely only on the selected-action SHA.
- Keep the marker coverage only in Project Director runtime output.
- Add a report-only note without a failing validation check.

## 6. Risks

The selected-action baseline becomes stricter, so stale manual edits to the
current command will fail Project Director/no-WING checks sooner. That is the
intended safety behavior for this rail.

## 7. Test plan

- Targeted no-WING handoff and summary tests.
- Targeted Project Director tests.
- no-WING quick check.
- Project Director CLI check.
- Focused WING/no-WING suite.
- Full pytest, memory lint, and whitespace diff check.

## 8. Where the other agent may disagree

Kimi may argue that selected-action SHA and baseline coverage already make this
redundant. The counterpoint is operator visibility: the value-gate marker check
should be visible at the same summary level as other no-WING handoff gates.
