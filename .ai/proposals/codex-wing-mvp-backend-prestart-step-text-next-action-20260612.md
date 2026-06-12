# Codex Proposal: WING MVP Backend Prestart Step Text Next Action

## Short Thesis

Expose the raw first operator step text in selected-action and runbook surfaces,
requiring it to mention `prestart_backend_port_check` and `unexpected
listeners`.

## What I Understood

The current WING MVP rail already requires the backend-port prestart check to
be the first operator step. The next useful hardening is making the operator
step content explicit, so reviewers can reject a handoff packet whose first
step is indexed correctly but points at the wrong action.

## Proposed Solution

- Add `backend_port_prestart_check_operator_sequence_step` to the first blocker
  selected action and current command.
- Require the step text to include `prestart_backend_port_check` and
  `unexpected listeners`.
- Mirror the requirement in the read-only runbook, MVP doc, autoloop prompt,
  and README.
- Add assertions in Project Director and WING MVP rail tests.

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

- Rely on `backend_port_prestart_gate` exact match only. Rejected because the
  raw step field is already emitted and easier for operator review.
- Add runtime behavior. Rejected because this slice should not contact WING or
  change live authority.

## Risks

The change is documentation/test-only. The risk is wording drift between the
selected action, runbook, and tests.

## Test Plan

- Focused WING rail tests.
- Project Director and direction guard JSON checks.
- Full pytest.

## Where The Other Agent May Disagree

Kimi may consider the exact gate match sufficient. The counterpoint is that the
operator-facing text should name the raw field humans inspect in the no-WING
handoff summary.
