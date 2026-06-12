# Proposal: Make WING handoff plan output explicit

## Short Thesis

The optional WING read-only handoff command should explicitly include the
expected-plan path used by the real preflight flow.

## What I Understood

The preflight runner is strict about the offline expected plan. Operator-facing
commands should make the plan path explicit everywhere a handoff references the
preflight sequence.

## Proposed Solution

- Add `--plan-output artifacts/wing_readonly_preflight_plan/latest_plan.json`
  to the runbook handoff command.
- Add a rails assertion covering that path.

## Likely Files To Touch

- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tests/test_wing_mvp_rails.py`

## Alternatives Considered

Relying on the handoff tool default was rejected because explicit runbook
commands are safer for strict expected-plan workflows.

## Risks

This is documentation/test-only and does not widen write authority.

## Test Plan

- Focused rails test.
- Adjacent WING/no-WING tests.
- Full pytest before commit.

## Where The Other Agent May Disagree

Kimi may consider the default enough. I prefer explicitness because the whole
preflight proof path depends on exact plan identity.
