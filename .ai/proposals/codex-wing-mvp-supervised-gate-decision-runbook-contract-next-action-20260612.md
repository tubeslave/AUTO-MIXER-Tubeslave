# Proposal: WING MVP Supervised Gate Decision Runbook Contract Next Action

Date: 2026-06-12

## 1. Short Thesis

Make the selected WING MVP next action explicitly require the structured
`supervised_prestart_gate` to stay tied to an accepted decision and only unlock
the supervised fader/gain proof runbook.

## 2. What I Understood

The handoff packet and validator already carry the structured fields. The
selected action should expose them because operators review the selected action
before any supervised proof planning.

## 3. Proposed Solution

- Add `supervised_prestart_gate.requires_operator_decision_status:
  accepted_decision_ready` and `supervised_prestart_gate.unlocks_runbook:
  Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md` to the first WING MVP
  blocker next action.
- Mirror the same fields in current first command, read-only runbook, MVP doc,
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

Leave the full-object equality requirement alone. Rejected because the
operator-facing selected action should surface the accepted-decision and runbook
boundary directly.

## 6. Risks

The main risk is wording drift between validator, runbook, and selected action.
Focused string tests reduce that risk.

## 7. Test Plan

- Focused WING rail tests.
- Direction guard and Project Director JSON checks.
- `git diff --check`.
- Full pytest before commit.

## 8. Where The Other Agent May Disagree

Kimi may argue full equality already proves this. The counterpoint is that this
change clarifies the operator review contract without widening live authority.
