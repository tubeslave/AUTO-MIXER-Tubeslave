# Proposal: WING MVP Blocker Handoff Anchor Next Action

Date: 2026-06-12

## Short thesis

Promote the `mvp_blocker_anchor` handoff requirement into the WING MVP blocker
next action so Project Director cannot omit it from the selected work directive.

## What I understood

The current MVP remains supervised WING Rack proof, and the first blocker is
WING read-only connection proof. Offline rails may improve proof discipline, but
they must not contact WING or claim readiness.

## Proposed solution

- Update the first blocker next action to require `mvp_blocker_anchor` matching
  the current first open row.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md` from the blocker table.
- Extend tests so Project Director, blocker parsing, and generated progress all
  preserve the anchor requirement.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/INDEX.md`

## Alternatives considered

- Leave the requirement only in the handoff validator. That catches stale
  packets, but Project Director would not surface the requirement in the next
  action text.
- Add a separate progress-only note. That is weaker than making the live blocker
  row the source of truth.

## Risks

- The first blocker next action becomes longer.
- If future blocker rows change, tests will force updating the generated report
  and Director expectations together.

## Test plan

- Focused Project Director and WING MVP rails tests.
- Regenerate progress report.
- Direction guard check.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may suggest keeping the next action shorter. I would keep the explicit
anchor text because the selected action is what future autonomous cycles read
before choosing work.
