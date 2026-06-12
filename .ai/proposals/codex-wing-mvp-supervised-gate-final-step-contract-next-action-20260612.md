# Proposal: WING MVP Supervised Gate Final-Step Contract Next Action

Date: 2026-06-12

## 1. Short Thesis

Make the selected WING MVP next action explicitly require the structured
`supervised_prestart_gate` to prove it is the final operator step before
supervised fader/gain proof planning.

## 2. What I Understood

The handoff packet and validator already carry a structured supervised prestart
contract. The selected action should not rely only on summary fields when the
gate object itself contains the authoritative final-step contract.

## 3. Proposed Solution

- Add `supervised_prestart_gate.operator_sequence_step_index: 9` and
  `supervised_prestart_gate.operator_sequence_step_must_be_final: true` to the
  first WING MVP blocker next action.
- Mirror the same fields in the current first command, read-only runbook, MVP
  doc, autoloop prompt, and README.
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
operator-facing next action should expose the field-level contract reviewers must
confirm before supervised proof planning.

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
