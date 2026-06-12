# Codex Proposal: WING MVP Backend Prestart Failure Detail Fields Next Action

## Short Thesis

Require the structured no-WING handoff to name the failure-detail fields needed
to review unexpected backend-port listeners safely.

## What I Understood

The WING MVP rail already requires the backend-port prestart check command to
run before the intended backend/WING session. The next useful hardening is
making the fail-closed detail field list explicit, so operators can reject a
handoff that does not expose listener processes, safe-to-stop status, failure
reason, and next-step evidence.

## Proposed Solution

- Add `backend_port_prestart_check_failure_detail_fields` to the first blocker
  selected action and current command.
- Require the field list to include listener process, confirmation checklist,
  blocked-until-confirmed, backend confirmation, safe-to-stop, fail-closed
  reason, and operator next-step fields.
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

- Rely on command matching only. Rejected because command identity does not
  prove the packet exposes the fields needed for operator review.
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

Kimi may consider the exact command sufficient. The counterpoint is that
fail-closed review needs the structured evidence fields, not only the command
that produced them.
