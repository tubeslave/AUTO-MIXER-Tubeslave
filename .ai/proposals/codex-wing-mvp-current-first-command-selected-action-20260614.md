# Codex Proposal: WING MVP Current First Command Selected Action

## Short thesis

Use `## Current First Command` as the selected action when it explicitly covers
the safety/proof baseline, and keep the baseline fallback for drift.

## What I understood

The WING read-only connection proof remains the active blocker. The selected
action became very large because Project Director appends the blocker table row
even after the current command was updated with the same proof-plan contract.

## Proposed solution

- Add a required marker list for `Current First Command` coverage.
- Return `Current First Command` directly only when all markers are present.
- Add the missing listener-template denial fields to `Current First Command` so
  the direct action does not lose those constraints.
- Keep the existing baseline append behavior when the marker check fails.

## Likely files to touch

- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `Docs/adr/wing-mvp-current-first-command-selected-action-20260614.md`

## Alternatives considered

1. Keep appending the full baseline forever.
2. Compare current-command text to the blocker row for exact equality.
3. Use a marker coverage contract and retain fallback on drift.

## Risks

The main risk is accidentally dropping a safety constraint from the selected
action. The marker list and existing tests reduce this risk, and fallback still
preserves the blocker row when coverage fails.

## Test plan

- Focused Project Director and WING rail tests.
- Regenerate MVP progress.
- Project Director JSON and Direction Guard.
- no-WING quick fail-closed check.
- memory lint, diff whitespace check, and full pytest suite.

## Where the other agent may disagree

Kimi may prefer continuing to append the blocker row because it is maximally
conservative. My recommendation keeps that behavior as fallback while reducing
routine Goal-mode prompt size only after explicit marker coverage passes.
