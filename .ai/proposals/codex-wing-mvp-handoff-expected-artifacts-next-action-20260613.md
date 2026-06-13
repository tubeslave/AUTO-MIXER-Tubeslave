# Codex Proposal: WING MVP Handoff Expected Artifacts Next Action

## Short thesis

Make the expected-artifacts handoff contract part of the canonical WING MVP
selected action so Project Director and progress reports keep enforcing it.

## What I understood

The current first blocker is still real WING read-only connection proof. The
last code change made the packet and no-WING validator compare handoff
artifacts against the expected plan, but the blocker table and cycle rails did
not yet require reviewers to look at those fields before supervised proof
planning.

## Proposed solution

Add the expected-artifacts checks to `.ai/project-layer/wing_mvp_blockers.md`,
`Docs/MVP_WING_SUPERVISED.md`, `Docs/runbooks/WING_READONLY_PREFLIGHT.md`,
`tools/codex_autoloop/README.md`, and
`tools/codex_autoloop/codex_cycle_prompt.md`. Regenerate the progress report and
extend rail tests so Project Director, progress output, runbook, prompt, and
docs index preserve the contract.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `Docs/INDEX.md`

## Alternatives considered

Leave the new fields only in no-WING summary output. Rejected because Goal-mode
work direction is selected from the blocker table, not from a human memory of a
prior packet change.

## Risks

The selected-action string is already large and this adds more duplication.
The mitigation is keeping the addition limited to the exact artifact drift
fields and covering the duplication with focused tests.

## Test plan

Run the progress updater, focused Project Director/WING rail tests, direction
guard, Project Director JSON output, `git diff --check`, and memory lint.

## Where Kimi may disagree

Kimi may prefer replacing the long selected-action string with a generated
contract. That is likely better later, but this scoped patch preserves the
current parser and avoids changing task-selection semantics.
