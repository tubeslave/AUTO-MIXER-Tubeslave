# Codex Proposal: WING MVP Read Only Required By Packet Next Action

## Short Thesis

Require the no-WING handoff summary to state that the packet is read-only.

## What I Understood

The WING MVP rail already denies packet execution, readiness updates, readiness
closure, and live writes. The next useful hardening is the positive read-only
field: a no-WING handoff packet can prepare review only if it identifies itself
as read-only.

## Proposed Solution

- Add `read_only: true` to the first blocker selected action and current
  command.
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

- Rely on `live_write: false`. Rejected because read-only identity is a
  separate positive packet claim and should be checked directly.
- Add runtime behavior. Rejected because this slice must not contact WING,
  stop listeners, or change live authority.

## Risks

The change is documentation/test-only. The main risk is wording drift between
selected action, runbook, generated progress, prompt, and README.

## Test Plan

- Focused WING rail tests.
- Project Director and direction guard JSON checks.
- Diff check, memory lint, and full pytest.

## Where The Other Agent May Disagree

Kimi may consider live-write denial enough. The counterpoint is that operator
review should also expose the packet-level read-only identity field.
