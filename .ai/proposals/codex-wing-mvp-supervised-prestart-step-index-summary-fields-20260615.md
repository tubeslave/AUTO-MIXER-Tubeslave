# Codex Proposal: WING MVP Supervised Prestart Step Index Summary Fields

## Short Thesis

Publish the supervised prestart required/final-step contract as flat no-WING
summary fields derived from the structured gate.

## What I Understood

The WING MVP remains blocked on real read-only proof. The offline rail should
make every pre-supervised-proof gate directly inspectable in release summaries.

## Proposed Solution

- Derive top-level fields for required, step index, and final-step status from
  `supervised_prestart_gate`.
- Render those fields in no-WING `summary.json` and `summary.md`.
- Require them in Project Director Current First Command baseline coverage.
- Keep all behavior no-authority and offline.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`

## Alternatives Considered

- Keep these fields nested only. Rejected because direct summary inspection is
  safer for operator review.
- Generate all nested projections from a registry. Deferred to avoid a larger
  refactor in the active WING rail.

## Risks

- Flat fields could drift from the structured gate. Mitigation: derive the
  fields and assert JSON/Markdown parity.

## Test Plan

- Compile touched Python files.
- Run focused no-WING and Project Director tests.
- Run no-WING quick validation.
- Run the full pytest suite.
- Run memory lint and diff whitespace check.

## Where The Other Agent May Disagree

Kimi may prefer centralizing all summary projections. The scoped projection is
lower risk for the current blocker.
