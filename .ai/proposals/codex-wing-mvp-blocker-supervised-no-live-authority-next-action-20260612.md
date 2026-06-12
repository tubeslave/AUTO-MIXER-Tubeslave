# Proposal: WING MVP Blocker Supervised No-Live Authority Next Action

## Short thesis

Make the first open WING MVP blocker next action explicitly require
`supervised_prestart_gate.grants_live_authority: false`.

## What I understood

The handoff packet, release summary, and runbook already treat the supervised
prestart gate as non-authorizing. Project Director consumes the blocker table,
so the table must state that field directly to keep the selected action as
strict as the validator.

## Proposed solution

- Update `.ai/project-layer/wing_mvp_blockers.md`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Add focused assertions in Project Director and WING MVP rail tests.
- Record the decision in ADR/report/index surfaces.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/adr/wing-mvp-blocker-supervised-no-live-authority-next-action-20260612.md`
- `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_NO_LIVE_AUTHORITY_NEXT_ACTION_20260612.md`

## Alternatives considered

- Leave the field only in the runbook and validator. Rejected because Project
  Director's selected action is the next-cycle entrypoint.
- Add a new runtime gate. Rejected because runtime validation already has the
  structured no-live field; this change is an alignment fix.

## Risks

Low. The change is text and tests only. The main risk is stale generated
progress, covered by regenerating `MVP_PROGRESS_LATEST.md`.

## Test plan

- Run focused Project Director and WING MVP rail tests.
- Run direction guard and full pytest before commit.

## Where the other agent may disagree

Kimi may argue this is redundant with the runbook. I would keep it because the
autoloop consumes the blocker table directly and should not rely on a later
document lookup for a live-authority boundary.
