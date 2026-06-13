# Codex Proposal: WING MVP Busy-Port Prestart Gate Authority Summary

## Short Thesis

Make the failed busy-port prestart artifact self-contained by adding explicit
non-authority markers to its detail, Markdown summary, shared failure-detail
contract, and WING MVP rails.

## What I Understood

The first open blocker remains `WING read-only connection proof`. Existing work
made the busy-port prestart gate visible and non-optional, but a failed
busy-port summary should directly state that the gate grants no execution,
readiness-closure, or live authority.

## Proposed Solution

Add these fields to failed busy-port `backend_port_free` detail:

- `backend_port_prestart_gate_execution_allowed: false`
- `backend_port_prestart_gate_readiness_closure_allowed: false`
- `backend_port_prestart_gate_live_authority_granted: false`

Render them in `summary.md`, include them in
`WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS`, and
require them in the blocker table, MVP doc, runbook, autoloop prompt/README, and
Project Director rails.

## Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_wing_mvp_rails.py`
- `tests/test_project_director_agent.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_GATE_AUTHORITY_SUMMARY_20260613.md`

## Alternatives Considered

1. Add only docs wording. Rejected because the first failed artifact would still
   omit machine-readable authority denial.
2. Reuse generic `live_authority_granted`. Rejected because the gate needs
   execution and readiness-closure denial too, and the field should be scoped to
   the prestart gate.
3. Add explicit prestart-gate authority fields. Chosen because it is narrow,
   testable, and aligns JSON, Markdown, and docs.

## Risks

- The shared failure-detail list is exact, so tests and docs must be updated
  together.
- Adding fields must not imply a passed no-WING check closes the WING blocker.

## Test Plan

- Compile touched Python modules.
- Run focused no-WING and WING MVP rail tests.
- Run a no-WING quick busy-port fail-closed check and inspect saved summary
  artifacts.
- Run Project Director and direction guard.
- Run full pytest before commit.

## Where The Other Agent May Disagree

Kimi may prefer keeping the failure-detail list stable and adding only top-level
fields. I think the exact list should include the fields because downstream
handoff and review plans use that list as the contract for what failed busy-port
summaries must expose.
