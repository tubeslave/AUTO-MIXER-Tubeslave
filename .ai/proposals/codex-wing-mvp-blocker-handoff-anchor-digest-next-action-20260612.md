# Proposal: WING MVP Blocker Handoff Anchor Digest Next Action

Date: 2026-06-12

## Short thesis

Promote the `mvp_blocker_anchor_sha256` handoff requirement into the WING MVP
blocker next action so Project Director cannot omit the digest check from the
selected work directive.

## What I understood

The current MVP remains supervised WING Rack proof, and the first blocker is
WING read-only connection proof. The project can harden offline rails before
hardware proof, but those rails must not contact WING, widen writes, or claim
live readiness.

## Proposed solution

- Update the first blocker next action to require
  `mvp_blocker_anchor_sha256_matches_current_first_open: true`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md` from the blocker table.
- Extend Project Director and WING MVP rails tests so the digest condition is
  preserved in selected action text and generated progress output.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/INDEX.md`

## Alternatives considered

- Leave the digest requirement only in the handoff validator. That fails stale
  packets, but future autonomous cycles would not see the condition while
  planning.
- Add a report-only note. That is weaker than making the live blocker row the
  source of truth.

## Risks

- The first blocker next action becomes longer.
- If the handoff summary field is renamed, tests will force updating blocker
  text, generated progress, and Project Director expectations together.

## Test plan

- Regenerate the latest MVP progress report.
- Focused Project Director, WING MVP rails, and direction guard tests.
- Direction guard and Project Director JSON smoke checks.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may prefer a shorter next action. I would keep the digest text explicit
because the selected action is the instruction future autonomous cycles read
before choosing work.
