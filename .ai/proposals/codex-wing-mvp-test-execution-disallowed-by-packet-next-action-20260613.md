# Proposal: Require `executes_tests: false` Before Supervised Proof Planning

## Short thesis

The no-WING handoff packet must be explicitly non-validating before it can guide
supervised proof planning.

## What I understood

The project is moving toward autonomous WING control, but the first open blocker
is still a real read-only WING connection proof. The offline handoff packet can
describe what an operator should do next, but it does not run tests and cannot
replace validation output from real commands.

## Proposed solution

Add `executes_tests: false` to the WING MVP next action, runbook validation,
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

Leave the packet field implicit. Rejected because future agents could read the
handoff packet as if it were validation output.

## Risks

The selected-action string is long and can become inconsistent across docs. The
mitigation is focused regression coverage and regenerated progress.

## Test plan

- Focused WING/autoloop pytest slice.
- Project Director JSON output check.
- Direction guard JSON output check.
- `git diff --check`.
- Memory lint and full pytest before commit.

## Where the other agent may disagree

Kimi may prefer consolidating packet flag requirements into generated docs. That
is reasonable later, but this narrow step preserves the current rails while
making the safety boundary explicit now.
