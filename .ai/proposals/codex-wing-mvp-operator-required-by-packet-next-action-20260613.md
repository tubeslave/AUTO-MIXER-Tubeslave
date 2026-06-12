# Proposal: Require `operator_required: true` Before Supervised Proof Planning

## Short thesis

The no-WING handoff packet must explicitly require a human operator before it
can guide real WING preflight or supervised proof planning.

## What I understood

The long-term goal is autonomous mixing, but the current first blocker is still
read-only WING connection proof. The offline handoff packet may describe the
next operator sequence, but it must not be interpreted as permission for
unsupervised execution.

## Proposed solution

Add `operator_required: true` to the WING MVP next action, runbook validation,
MVP stop rules, autoloop README/prompt, generated progress, and regression
tests.

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

Leave the field implicit in the packet JSON. Rejected because future handoff
review should not infer operator gating from surrounding prose.

## Risks

The selected-action string is already long and may become harder to maintain.
The mitigation is focused tests and regenerated progress.

## Test plan

- Focused WING/autoloop pytest slice.
- Project Director JSON output check.
- Direction guard JSON output check.
- `git diff --check`.
- Memory lint and full pytest before commit.

## Where the other agent may disagree

Kimi may prefer generating all packet flag rails from the packet builder. That
would reduce duplication later, but this narrow patch keeps the current
operator-facing contract explicit now.
