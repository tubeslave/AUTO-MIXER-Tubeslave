# Codex Proposal: WING MVP Backend Prestart Required Markers Next Action

## Short Thesis

Require the structured backend-port prestart gate to expose the exact markers
that make the first operator step reviewable.

## What I Understood

The WING MVP rail already requires the backend-port prestart gate to be first,
non-authoritative, and bound to `prestart_backend_port_check`. The next useful
hardening is requiring the gate marker list to name the session boundary, the
command, and the unexpected-listener condition.

## Proposed Solution

- Add `backend_port_prestart_gate.required_markers` to the first blocker
  selected action and current command.
- Require the marker list to include `Before starting the intended backend/WING
  session`, `prestart_backend_port_check`, and `unexpected listeners`.
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

- Rely on the raw operator step text only. Rejected because the structured gate
  already carries the exact marker list and can be asserted.
- Add runtime behavior. Rejected because this slice should not contact WING or
  change live authority.

## Risks

The change is documentation/test-only. The main risk is wording drift between
selected action, runbook, generated progress, prompt, and README.

## Test Plan

- Focused WING rail tests.
- Project Director and direction guard JSON checks.
- Diff check, memory lint, and full pytest.

## Where The Other Agent May Disagree

Kimi may consider the command binding sufficient. The counterpoint is that
operator review depends on the full stop condition, not only the command name.
