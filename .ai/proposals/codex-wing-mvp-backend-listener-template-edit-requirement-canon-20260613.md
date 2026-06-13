# Proposal: WING MVP Backend Listener Template Edit Requirement Canon

## Short thesis

Make the blocker table and Project Director carry the same template-edit
acceptance boundary that the no-WING validator already enforces.

## What I understood

The active MVP remains the first WING read-only connection proof. The latest
validator change prevents placeholder-filled generated templates from clearing
the backend-port prestart gate, but the canonical next action should also tell
future Codex cycles to preserve that boundary.

## Proposed solution

- Add the template edit gate report to the first blocker evidence list.
- Add required edit markers and template-derived acceptance markers to the
  first blocker next action.
- Regenerate `MVP_PROGRESS_LATEST.md` from the blocker table.
- Update `Docs/MVP_WING_SUPERVISED.md`, the Codex autoloop prompt/README, and
  tests that verify Project Director output.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_wing_mvp_rails.py`
- `tests/test_project_director_agent.py`

## Alternatives considered

Leaving the runbook as the only source is weaker: Project Director would still
summarize the older pending-template-only contract. Adding more validator code
is unnecessary because the behavior is already enforced.

## Risks

The next-action text grows longer, but this repo already uses explicit
operator-visible gate contracts for the WING MVP. The benefit is that future
goal-mode cycles receive the stricter requirement directly.

## Test plan

Run rail tests, Project Director tests, direction guard, memory lint, and the
standard full pytest command before commit.

## Where the other agent may disagree

Kimi may prefer moving the long next-action contract into a structured schema.
That is reasonable later; for this step, preserving the established blocker
table flow is lower risk.
