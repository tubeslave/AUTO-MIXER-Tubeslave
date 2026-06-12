# Proposal: Require `queries_console: false` Before Supervised Proof Planning

## Short thesis

The no-WING handoff packet must be explicitly non-querying before it can inform
supervised WING proof planning.

## What I understood

The long-term goal is autonomous console control, but the current first blocker
is still read-only WING connection proof. The packet builder already emits
`queries_console: false`, meaning the offline packet does not prove a console was
queried. The selected action should make that boundary visible to operators and
future agents.

## Proposed solution

Add `queries_console: false` to the WING MVP next action, runbook validation,
MVP stop rules, autoloop README/prompt, generated progress, and regression
tests. Keep the packet as non-executing offline handoff evidence only.

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

Leave the packet field implicit. Rejected because implicit no-query semantics
are easy to miss when the Project Director selected action is used as the
operator handoff.

## Risks

The main risk is another long selected-action string becoming inconsistent
across docs and tests. The mitigation is focused regression coverage and
regenerated progress.

## Test plan

- Focused WING/autoloop pytest slice.
- Project Director JSON output check.
- Direction guard JSON output check.
- `git diff --check`.
- Memory lint and full pytest before commit.

## Where the other agent may disagree

Kimi may prefer consolidating these packet flags into a generated contract table
instead of repeating them in docs. That is reasonable later, but the smallest
safe step is to make the current operator-facing rails enforce the field now.
