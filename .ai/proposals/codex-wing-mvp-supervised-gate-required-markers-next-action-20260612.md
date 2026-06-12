# Proposal: WING MVP Supervised Gate Required Markers Next Action

Date: 2026-06-12

## 1. Short Thesis

Make the selected WING MVP next action explicitly require the structured
`required_supervised_prestart_gate_markers` list before supervised proof
planning.

## 2. What I Understood

The handoff packet and validator already expose the marker list. Operators and
Project Director should see that exact field in the selected action instead of
only seeing the marker strings embedded in a sentence.

## 3. Proposed Solution

- Add `required_supervised_prestart_gate_markers` and its three expected markers
  to the first WING MVP blocker next action.
- Mirror the same field in current first command, read-only runbook, MVP doc,
  autoloop prompt, and README.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Add focused tests for Project Director, blocker parsing, runbook text,
  autoloop docs, MVP docs, and generated progress.

## 4. Likely Files To Touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`

## 5. Alternatives Considered

Leave the marker strings only in the final step text. Rejected because the
handoff summary exposes a structured marker-list field that reviewers can check
directly.

## 6. Risks

The main risk is wording drift between validator, runbook, and selected action.
Focused string tests reduce that risk.

## 7. Test Plan

- Focused WING rail tests.
- Direction guard and Project Director JSON checks.
- `git diff --check`.
- Full pytest before commit.

## 8. Where The Other Agent May Disagree

Kimi may consider this redundant because the final step text carries the same
markers. The counterpoint is that field-level review is less ambiguous.
