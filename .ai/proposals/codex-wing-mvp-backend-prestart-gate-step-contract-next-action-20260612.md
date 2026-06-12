# Proposal: WING MVP Backend Prestart Gate Step Contract Next Action

Date: 2026-06-12

## 1. Short Thesis

Make the selected WING MVP next action explicitly require the structured
`backend_port_prestart_gate` to prove it is the first operator step.

## 2. What I Understood

The handoff packet and validator already carry a structured backend-port
prestart contract. The selected action should not rely only on summary fields
when the gate object itself contains the authoritative first-step contract.

## 3. Proposed Solution

- Add `backend_port_prestart_gate.operator_sequence_step_index: 0` and
  `backend_port_prestart_gate.operator_sequence_step_must_be_first: true` to the
  first WING MVP blocker next action.
- Mirror the same fields in the current first command, autoloop prompt, README,
  and MVP stop/current-action text.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Add focused tests for Project Director, blocker parsing, autoloop docs, MVP
  docs, and generated progress.

## 4. Likely Files To Touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`

## 5. Alternatives Considered

Leave the equality check alone because it implies the fields. Rejected because
operator-facing selected actions should name the field-level contract that must
be reviewed before supervised proof planning.

## 6. Risks

The main risk is wording drift between the validator, runbook, and selected
action. Focused string tests reduce that risk.

## 7. Test Plan

- Focused WING rail tests.
- Direction guard and Project Director JSON checks.
- `git diff --check`.
- Full pytest before commit.

## 8. Where The Other Agent May Disagree

Kimi may argue that exact gate equality already proves this. The counterpoint is
that the next-action text is an operator contract, not just a machine check.
