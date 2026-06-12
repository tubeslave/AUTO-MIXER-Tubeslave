# Proposal: WING MVP Blocker Handoff Digest Review Next Action

Date: 2026-06-12

## Short thesis

Promote the handoff packet digest-review operator-sequence requirement into the
WING MVP blocker next action.

## What I understood

The current MVP remains supervised WING Rack proof, blocked on real read-only
connection evidence. Offline work should harden the next handoff without
contacting WING or widening writes.

## Proposed solution

- Update the first blocker next action to require
  `operator_sequence_has_mvp_blocker_anchor_digest_review: true`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Extend Project Director and WING MVP rails tests to preserve the condition.
- Record the decision and validation report.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/INDEX.md`

## Alternatives considered

- Leave the condition only in the handoff validator. That catches bad packets,
  but Project Director would not surface the requirement in its selected action.
- Add only a runbook note. The runbook is useful, but the blocker table remains
  the live task source.

## Risks

- The first blocker next action grows longer.
- If the summary field is renamed, tests will force updating the blocker table
  and generated progress together.

## Test plan

- Regenerate latest MVP progress.
- Focused Project Director, WING MVP rails, and direction guard tests.
- Project Director and direction guard JSON smoke checks.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may prefer keeping the blocker row shorter. I would keep this condition in
the live blocker row because future autonomous cycles read it before choosing
work.
