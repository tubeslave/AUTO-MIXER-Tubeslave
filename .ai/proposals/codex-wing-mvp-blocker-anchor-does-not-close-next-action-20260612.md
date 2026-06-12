# Proposal: WING MVP Blocker Anchor Does Not Close Next Action

## Short thesis

Make the WING selected action explicitly require
`mvp_blocker_anchor.packet_does_not_close_blocker: true`.

## What I understood

Offline handoff packets can guide operators, but they must not be mistaken for
real WING proof or blocker closure. The packet already carries a structured
field for that boundary; Project Director should surface it in the selected
action.

## Proposed solution

- Update `.ai/project-layer/wing_mvp_blockers.md`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Add focused assertions in Project Director and WING MVP rail tests.
- Record the change in ADR/report/index surfaces.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/adr/wing-mvp-blocker-anchor-does-not-close-next-action-20260612.md`
- `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_ANCHOR_DOES_NOT_CLOSE_NEXT_ACTION_20260612.md`

## Alternatives considered

- Leave the field only in the runbook. Rejected because the autoloop consumes
  the blocker table before reviewing runbook detail.
- Add new validator behavior. Rejected because the validator already fails
  closed if the packet anchor tries to close the blocker.

## Risks

Low. This is a text/test sync. The main risk is stale generated progress,
covered by regenerating it from the blocker table.

## Test plan

- Focused Project Director, WING MVP rails, and direction guard tests.
- Direction guard CLI.
- `git diff --check`, memory lint, and full pytest before commit.

## Where the other agent may disagree

Kimi may view the requirement as redundant with runbook text. I would keep it
because the selected action is the autonomous cycle entrypoint and should carry
the offline-evidence boundary directly.
