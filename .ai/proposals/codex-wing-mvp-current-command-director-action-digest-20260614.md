# Codex Proposal: WING MVP Current Command Director Action Digest

## Short thesis

Promote the handoff packet's Project Director selected-action digest into the
current Project Director action contract.

## What I understood

The active blocker remains `WING read-only connection proof`. The previous step
made the handoff packet carry the selected-action anchor and SHA256, but the
`Current First Command` still did not require those no-WING handoff summary
fields. Autonomous cycles should not lose that requirement.

## Proposed solution

Add the `project_director_selected_action_*` and baseline coverage verdict
fields to `## Current First Command`, add the same markers to
`CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS`, document them in the autoloop
prompt/README, and extend Project Director/progress tests.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/update_mvp_progress.py`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- Council brief, ADR, report, and index files

## Alternatives considered

Leaving the requirement only in the handoff packet validator is safe for packet
validation but weak for Goal-mode direction. Updating the first blocker table
row's long fallback action was deferred because `Current First Command` is the
selected action source and the baseline marker set now proves it covers the new
digest fields.

## Risks

The main risk is wording drift in the long WING rail text. The focused tests
check both selected-action text and progress output, and the no-WING packet
schema still validates the actual fields.

## Test plan

Run focused Project Director, WING rail, autoloop prompt, and autoloop contract
tests. Then run Project Director, direction guard, no-WING quick fail-closed,
memory lint, `git diff --check`, and the standard full pytest command.

## Where the other agent may disagree

Kimi may prefer editing the first blocker table row at the same time. The
counterpoint is that Project Director is currently using `Current First
Command`, which is now covered by explicit baseline markers and regression
tests; changing the already very long table row is higher churn.
