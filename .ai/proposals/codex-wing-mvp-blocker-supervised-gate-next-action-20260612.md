# Proposal: WING MVP Blocker Supervised Gate Next Action

Date: 2026-06-12

## Short thesis

Promote the supervised prestart gate parity requirements into the first WING MVP
blocker next action.

## What I understood

The project is still waiting on real WING read-only proof. Offline work should
make the next handoff safer and more explicit, without contacting WING or
expanding live-write authority.

## Proposed solution

- Update the first blocker next action to require
  `operator_sequence_supervised_prestart_gate_is_final: true`.
- Require `supervised_prestart_gate` to exactly match
  `expected_supervised_prestart_gate`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Extend Project Director and WING MVP rails tests.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/INDEX.md`

## Alternatives considered

- Leave the condition only in the no-WING validator. That catches invalid
  packets later but does not guide Project Director.
- Keep the runbook as the only prose source. That is weaker than updating the
  live blocker row.

## Risks

- The first blocker next action gets longer.
- Future wording changes must update tests and generated progress together.

## Test plan

- Regenerate latest MVP progress.
- Focused Project Director, WING MVP rails, and direction guard tests.
- Project Director and direction guard JSON smoke checks.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may prefer shorter selected-action text. I would keep the supervised gate
conditions explicit because they are the final handoff boundary before any
supervised proof runbook can be used.
