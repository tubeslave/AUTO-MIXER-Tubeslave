# Proposal: Require Existing Backend Connection Field Before Supervised Proof Planning

## Short thesis

The no-WING handoff packet must explicitly require an already-connected backend
before any physical read-only preflight can support supervised proof planning.

## What I understood

The long-term goal is autonomous console control, but the current blocker is
still real read-only WING connection proof. The packet already says the physical
preflight requires an existing backend connection; selected-action rails should
make that explicit so future agents do not treat the packet as connect authority.

## Proposed solution

Add `physical_preflight_requires_existing_backend_connection: true` to the WING
MVP next action, runbook validation, MVP stop rules, autoloop README/prompt,
generated progress, and regression tests.

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

Rely on the existing read-only preflight runbook prose. Rejected because Project
Director selected work should carry the packet field literally.

## Risks

The selected-action string is long and can drift. The mitigation is focused
tests plus regenerated progress.

## Test plan

- Focused WING/autoloop pytest slice.
- Project Director JSON output check.
- Direction guard JSON output check.
- `git diff --check`.
- Memory lint and full pytest before commit.

## Where the other agent may disagree

Kimi may prefer generating this from the packet schema. That is reasonable
later, but this patch keeps the current operator-facing rail explicit now.
