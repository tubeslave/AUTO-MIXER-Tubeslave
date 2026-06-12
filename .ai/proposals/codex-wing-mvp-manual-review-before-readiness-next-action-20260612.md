# Codex Proposal: WING MVP Manual Review Before Readiness Next Action

## Short Thesis

Require the no-WING handoff summary to state that manual review is required
before any readiness update.

## What I Understood

The WING MVP rail already requires no-WING prestart checks and fail-closed
fields before the intended backend/WING session. The next useful hardening is
making the readiness boundary explicit: offline/no-WING handoff packets can
prepare review, but cannot update readiness without manual real-artifact
review.

## Proposed Solution

- Add `manual_review_required_before_readiness_update: true` to the first
  blocker selected action and current command.
- Mirror the requirement in the read-only runbook, MVP doc, autoloop prompt,
  and README.
- Add assertions in Project Director and WING MVP rail tests.
- Regenerate latest MVP progress and record validation in a report.

## Likely Files To Touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `Docs/INDEX.md`

## Alternatives Considered

- Rely on general readiness prose. Rejected because the handoff packet already
  has an exact structured field.
- Add runtime behavior. Rejected because this slice should not contact WING,
  stop listeners, or change live authority.

## Risks

The change is documentation/test-only. The main risk is wording drift between
selected action, runbook, generated progress, prompt, and README.

## Test Plan

- Focused WING rail tests.
- Project Director and direction guard JSON checks.
- Diff check, memory lint, and full pytest.

## Where The Other Agent May Disagree

Kimi may consider the existing blocker status enough. The counterpoint is that
operator handoff review should expose the explicit field that prevents
readiness updates from no-WING evidence.
