# Codex Proposal: WING MVP Plan Output Contract

## Short Thesis

The WING read-only preflight expected plan path should be a shared contract
value, and no-WING should validate every command that references it.

## What I Understood

The current MVP is blocked on real WING read-only proof. Before that run, the
operator must generate and use the expected offline plan. If the plan output
path drifts between preview, capture, and handoff commands, the operator can
start from stale proof instructions without any live mixer write being involved.

## Proposed Solution

- Add `WING_READONLY_PREFLIGHT_PLAN_OUTPUT` to the shared preflight contract.
- Make the handoff builder and no-WING release-check consume that value.
- Validate that the plan command `--plan-output`, capture command
  `--expected-plan`, and handoff packet command `--plan-output` match the shared
  contract.
- Surface contract-match booleans in JSON and Markdown summaries.
- Cover valid and stale handoff-plan cases in tests.

## Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-plan-output-contract-20260613.md`
- `.ai/proposals/codex-wing-mvp-plan-output-contract-20260613.md`
- `Docs/adr/wing-mvp-plan-output-contract-20260613.md`
- `Docs/reports/AUTOMIXER_WING_MVP_PLAN_OUTPUT_CONTRACT_20260613.md`

## Alternatives Considered

- Keep literals in place. Rejected because this leaves the expected-plan path
  vulnerable to silent drift.
- Centralize without validation. Rejected because stale handoff/capture command
  values could still pass no-WING.
- Change the path. Rejected because the current operator-facing path is already
  documented and should remain stable.

## Risks

The main risk is stale test fixtures. Runtime risk is low because the change is
limited to offline command construction and no-WING validation.

## Test Plan

- Compile touched modules.
- Run focused handoff, no-WING, Project Director, direction guard, and rails tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run `git diff --check`, memory lint, and the full pytest suite.

## Where The Other Agent May Disagree

Kimi may prefer centralizing URL, channel, and artifact-root defaults in the
same pass. This proposal keeps the diff focused on the expected plan output
path because it is directly tied to the preflight proof chain.
