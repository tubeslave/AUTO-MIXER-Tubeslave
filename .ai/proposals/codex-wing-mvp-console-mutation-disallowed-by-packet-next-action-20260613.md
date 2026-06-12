# Codex Proposal: WING MVP Console Mutation Disallowed By Packet Next Action

## Short Thesis

Require the no-WING handoff summary to state that the packet does not mutate
console state.

## What I Understood

The WING MVP rail already requires read-only identity and denies live writes.
The next useful hardening is the explicit console-mutation field: a no-WING
handoff packet can prepare review, but it must not be treated as evidence that
any mixer state changed.

## Proposed Solution

- Add `mutates_console_state: false` to the first blocker selected action and
  current command.
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

- Rely on `read_only: true` and `live_write: false`. Rejected because mutation
  status is a separate packet-level safety boundary and should be checked
  directly.
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

Kimi may consider read-only identity enough. The counterpoint is that operator
review should also expose the packet-level console-mutation denial field.
