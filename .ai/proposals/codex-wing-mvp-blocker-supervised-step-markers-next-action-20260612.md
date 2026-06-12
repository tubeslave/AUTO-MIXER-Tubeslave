# Proposal: WING MVP Blocker Supervised Step Markers Next Action

## Short thesis

Make Project Director's selected WING action require the same supervised
operator-step markers that the handoff validator already enforces.

## What I understood

The final supervised handoff operator step is the transition point from
accepted read-only review to supervised proof planning. It must name the
accepted decision, valid accepted-candidate prestart trace, and supervised proof
runbook before a fader/gain proof can proceed.

## Proposed solution

- Update `.ai/project-layer/wing_mvp_blockers.md` so the first blocker next
  action requires `operator_sequence_supervised_prestart_gate_step` to include
  the three marker strings.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Add focused Project Director and MVP rail assertions.
- Record the decision in ADR/report/index surfaces.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/adr/wing-mvp-blocker-supervised-step-markers-next-action-20260612.md`
- `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_STEP_MARKERS_NEXT_ACTION_20260612.md`

## Alternatives considered

- Leave the marker requirement only in the validator. Rejected because Project
  Director's next action is the autonomous cycle entrypoint.
- Add a new runtime proof step. Rejected because this cycle only needs to align
  selected-action text with existing validation.

## Risks

Low. This is a text/test sync. The risk is stale generated progress, mitigated
by regenerating it from the blocker table.

## Test plan

- Focused Project Director, WING MVP rails, and direction guard tests.
- Direction guard CLI.
- `git diff --check`, memory lint, and full pytest before commit.

## Where the other agent may disagree

Kimi may call it redundant with the runbook. I would keep it because the
autoloop consumes the blocker table before reading proof runbooks.
