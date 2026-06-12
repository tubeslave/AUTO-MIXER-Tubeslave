# Proposal: Require Expected Plan Field Before Supervised Proof Planning

## Short thesis

The no-WING handoff packet must explicitly require an offline expected plan
before any real WING read-only preflight can support supervised proof planning.

## What I understood

The project is moving toward autonomous WING control, but the first open blocker
is still real read-only connection proof. The handoff packet already says a real
run requires an expected plan; selected-action rails should make that field
visible so future agents do not treat ad hoc real runs as acceptable proof.

## Proposed solution

Add `expected_plan_required_for_real_run: true` to the WING MVP next action,
runbook validation, MVP stop rules, autoloop README/prompt, generated progress,
and regression tests.

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

Rely on the existing runbook prose about `--expected-plan`. Rejected because the
selected action is the primary Project Director handoff and should name the
packet field directly.

## Risks

The selected-action text is long and can drift. The mitigation is focused tests
plus regenerated progress.

## Test plan

- Focused WING/autoloop pytest slice.
- Project Director JSON output check.
- Direction guard JSON output check.
- `git diff --check`.
- Memory lint and full pytest before commit.

## Where the other agent may disagree

Kimi may prefer consolidating these packet flags into generated docs. That is
reasonable later, but this narrow patch keeps the current operator-facing rails
explicit now.
