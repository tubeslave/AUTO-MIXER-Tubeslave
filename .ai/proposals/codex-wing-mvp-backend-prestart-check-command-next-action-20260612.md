# Codex Proposal: WING MVP Backend Prestart Check Command Next Action

## Short Thesis

Require the structured no-WING handoff command field to match the backend-port
prestart release check command exactly.

## What I Understood

The WING MVP rail already requires the backend-port prestart gate to be first,
non-authoritative, marker-bound, and tied to `prestart_backend_port_check`.
The next useful hardening is requiring the packet's structured command field to
match the exact no-WING release check invocation.

## Proposed Solution

- Add `backend_port_prestart_check_command` to the first blocker selected
  action and current command.
- Require it to match `PYTHONPATH=backend python3 tools/no_wing_release_check.py
  --quick`.
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

- Rely on prose command text at the start of the selected action. Rejected
  because the structured handoff field should be explicit and assertable.
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

Kimi may consider the existing prose command sufficient. The counterpoint is
that operator packet review should inspect the structured command field, not
infer command identity from prose.
