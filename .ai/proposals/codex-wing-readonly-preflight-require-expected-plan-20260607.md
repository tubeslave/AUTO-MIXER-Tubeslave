# Proposal: Require Expected Plan For Real WING Read-Only Preflight

## Short Thesis

The runner should fail closed before WebSocket whenever a real read-only
preflight omits `--expected-plan`.

## What I Understood

The current MVP demands real WING evidence, but only through a bounded first
step. The offline plan is the operator-approved contract for URL, channel scope,
value types, artifact root, request sequence, review commands, and fail-closed
conditions.

## Proposed Solution

Add a pre-transport guard in `tools/wing_readonly_preflight_runner.py` after
basic argument validation and before expected-plan loading or WebSocket connect.
If `args.expected_plan` is empty, write a blocked local summary with
`reason: expected_plan_missing`, `expected_plan_match: false`, and
`expected_plan_violations: ["expected_plan_missing"]`, then exit `2`.

## Likely Files To Touch

- `tools/wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-readonly-preflight-require-expected-plan-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REQUIRE_EXPECTED_PLAN_20260607.md`
- `Docs/INDEX.md`

## Alternatives Considered

- Keep the runner permissive and rely on evidence intake rejection. Rejected
  because the real hardware contact point should enforce the runbook boundary.
- Make `argparse` require the flag globally. Rejected because `--plan-only`
  intentionally creates the expected plan and must not require one.

## Risks

- Existing manual invocations without `--expected-plan` will now fail. That is
  intended and safer.
- Tests that simulate real WebSocket runs must now include a generated expected
  plan.

## Test Plan

- Focused runner tests.
- no-WING quick gate.
- Direction guard.
- Memory lint.
- Full pytest before commit if time allows.

## Where The Other Agent May Disagree

Kimi may argue for enforcing this in `argparse`. The narrower pre-transport
guard is preferable because it preserves `--plan-only` ergonomics and produces a
local blocked summary artifact for audit.
