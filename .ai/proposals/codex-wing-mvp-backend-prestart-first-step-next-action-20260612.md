# Proposal: WING MVP Backend Prestart First-Step Next Action

## Short thesis

Make Project Director require the backend-port prestart check to be the first
operator step before any intended backend/WING session.

## What I understood

The handoff validator already exposes first-step fields for the backend-port
prestart check. Selected action and autoloop docs should carry those fields so
future cycles cannot skip the prestart check before backend/WING work.

## Proposed solution

- Update `.ai/project-layer/wing_mvp_blockers.md`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Update autoloop prompt and README.
- Add focused assertions in Project Director and WING MVP rail tests.
- Record the decision in ADR/report/index surfaces.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/adr/wing-mvp-backend-prestart-first-step-next-action-20260612.md`
- `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_FIRST_STEP_NEXT_ACTION_20260612.md`

## Alternatives considered

- Leave the fields only in the runbook. Rejected because the selected action is
  the autonomous cycle entrypoint.
- Add a new runtime gate. Rejected because existing handoff validation already
  fails closed when the first-step fields drift.

## Risks

Low. This is text/test synchronization. The main risk is stale generated
progress, covered by regenerating it.

## Test plan

- Focused Project Director, WING MVP rails, and direction guard tests.
- Project Director and direction guard CLIs.
- `git diff --check`, memory lint, and full pytest before commit.

## Where the other agent may disagree

Kimi may call this redundant with runbook text. I would keep it because the
selected action should be as strict as the handoff contract before any future
backend/WING session.
