# Proposal: Bind handoff prestart summary commands to the plan

## Short Thesis

The no-WING handoff summary should prove that preview and capture commands use
the same passed prestart `summary.json` required by the validated plan.

## What I Understood

The first MVP blocker is still `WING read-only connection proof`. The handoff
packet must prepare an operator for real read-only capture without granting live
authority or closing readiness from offline evidence. A stale packet should not
be able to point preview or capture at a different prestart summary.

## Proposed Solution

- Read the expected prestart summary path from the validated plan artifact.
- Extract preview and capture `--prestart-summary` command values.
- Fail validation when either command differs from the plan path.
- Emit JSON and Markdown summary fields for both command match booleans.
- Update WING MVP rails/docs so future cycles keep the requirement visible.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`

## Alternatives Considered

Relying on whole-command equality was rejected because the operator summary
needs the safety property visible as a direct boolean. Adding real WING checks
was rejected because this slice must remain no-WING.

## Risks

Hand-edited stale packets can fail until regenerated. That is intended because
the prestart summary is the boundary that proves no backend/WING contact
happened before the real read-only session.

## Test Plan

- Compile touched Python.
- Focused no-WING handoff tests.
- Project Director/WING rail tests.
- no-WING quick release check.
- Full pytest before commit.

## Where The Other Agent May Disagree

Kimi may argue the existing command equality was sufficient. I recommend the
explicit booleans because they make the operator-facing handoff auditable.
