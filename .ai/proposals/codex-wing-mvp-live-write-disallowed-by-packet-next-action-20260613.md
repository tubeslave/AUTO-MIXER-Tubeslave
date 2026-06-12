# Codex Proposal: WING MVP Live Write Disallowed By Packet Next Action

## Short Thesis

Require the no-WING handoff summary to state that the packet is not a live-write
artifact.

## What I Understood

The WING MVP rail already denies packet execution, readiness updates, and
readiness closure. The next useful hardening is the explicit live-write field:
a no-WING handoff packet can prepare review, but it must not be treated as any
kind of live-write proof.

## Proposed Solution

- Add `live_write: false` to the first blocker selected action and current
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

- Rely on `execution_allowed_by_packet: false`. Rejected because live-write
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

Kimi may consider the execution-denial field enough. The counterpoint is that
operator review should also expose the packet-level live-write denial field.
