# Proposal: WING MVP Blocker Supervised Trace Gate Next Action

Date: 2026-06-12

## Short thesis

Promote the supervised prestart trace gate fields into the first WING MVP
blocker next action.

## What I understood

The project is still blocked on real WING read-only connection proof. Offline
handoff hardening should make the next supervised proof path safer without
contacting WING or widening live writes.

## Proposed solution

- Update the first blocker next action to require
  `operator_sequence_has_supervised_prestart_gate: true`.
- Require
  `supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Extend Project Director and WING MVP rails tests.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/INDEX.md`

## Alternatives considered

- Leave the fields only in no-WING summaries. That is valid for artifact
  validation, but Project Director would not carry the full next-action
  boundary.
- Add only runbook prose. That does not update the live blocker table.

## Risks

- The selected action grows longer.
- If the summary field names change, tests will force updating blocker text and
  generated progress together.

## Test plan

- Regenerate latest MVP progress.
- Focused Project Director, WING MVP rails, and direction guard tests.
- Project Director and direction guard JSON smoke checks.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may prefer shorter blocker text. I would keep these explicit because they
describe the accepted-candidate prestart boundary that prevents supervised proof
planning from outrunning read-only review.
