# Codex Proposal: WING MVP Current First Command Proof-Plan Summary Fields

## Short thesis

Make the current first command carry the same proof-plan summary field contract
as the blocker row and no-WING handoff gate.

## What I understood

Project Director chooses the first non-closed MVP blocker and combines
`## Current First Command` with the blocker row baseline. The row already names
`required_supervised_proof_plan_summary_fields`, but the first command itself
does not. That makes the primary selected-action source weaker than the
baseline it is paired with.

## Proposed solution

Add the required supervised proof-plan summary field requirements to
`.ai/project-layer/wing_mvp_blockers.md` under `## Current First Command`.
Extend tests so `current_first_command()` and Project Director selected actions
must include the declared list, expected list, contract-match flag, expected
source IDs, and expected violations map.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `tests/test_wing_mvp_rails.py`
- `tests/test_project_director_agent.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `Docs/INDEX.md`
- `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_FIRST_COMMAND_PROOF_PLAN_SUMMARY_FIELDS_20260614.md`

## Alternatives considered

Rely on the blocker row baseline only. Rejected because Project Director names
`Current First Command` as the primary action source.

Only update tests. Rejected because tests must protect an actual operator-
visible command contract.

## Risks

The risk is another long first-command section. The field list is already part
of the accepted handoff/no-WING contract, and keeping the first command
self-contained reduces drift.

## Test plan

Run focused Project Director/WING rail tests, Project Director, direction
guard, no-WING quick fail-closed, memory lint, `git diff --check`, and the
standard full pytest command.

## Where the other agent may disagree

Kimi may prefer generating `Current First Command` from structured constants
instead of maintaining long Markdown. That is a useful follow-up, but this
patch fixes the immediate drift with a minimal diff.
