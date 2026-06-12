# Codex Proposal: WING MVP Backend Prestart Runs Command Next Action

## Short Thesis

Require the structured backend-port prestart gate to name the command it runs:
`prestart_backend_port_check`.

## What I Understood

The current WING MVP rail requires the backend-port prestart gate to be first
and non-authoritative before supervised proof planning. The next useful
hardening is binding that structured gate to the actual no-WING diagnostic
command, so a packet cannot satisfy the gate with an ambiguous or wrong
prestart action.

## Proposed Solution

- Add `backend_port_prestart_gate.runs_command: prestart_backend_port_check` to
  the first blocker selected action and current command.
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

- Rely on first-step index and first-step text only. Rejected because the
  structured gate should also identify the diagnostic command it is gating.
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

Kimi may consider the operator step text sufficient. The counterpoint is that
the structured gate is what reviewers scan for proof readiness, so it should
carry the command binding directly.
