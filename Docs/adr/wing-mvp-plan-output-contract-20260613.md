# ADR: WING MVP Plan Output Contract

## Context

The current MVP remains blocked on real `WING read-only connection proof`.
Before a real read-only preflight run, the operator must generate an offline
expected plan and then capture the physical WING readbacks against that plan.

The expected plan output path was duplicated between handoff/no-WING tooling.

## Options Considered

1. Keep the expected plan output path duplicated.
2. Move the path into the shared WING preflight contract and validate all
   command references to it.
3. Only change the handoff builder default.

## Decision

Use option 2. The shared preflight contract now owns
`WING_READONLY_PREFLIGHT_PLAN_OUTPUT`, and no-WING evidence validation fails
closed when the plan command, capture expected-plan command, or handoff command
points at a stale plan path.

## Why This Won

The expected plan is the offline artifact the operator uses to keep the real
WING preflight bounded and reviewable. Its path must be stable before a real
backend/WING session starts.

## Rejected Alternatives

Duplicated literals were rejected because they allow silent drift. Builder-only
centralization was rejected because no-WING still needs to reject stale command
values before a hardware step.

## Implementation Plan

- Add the expected plan output constant to the shared contract.
- Make handoff builder and no-WING consume the shared constant.
- Add no-WING detail/summary fields for plan path contract matches.
- Fail closed on stale plan, capture expected-plan, or handoff plan-output paths.
- Cover the happy path and stale handoff plan-output path in focused tests.

## Test Plan

- Compile touched modules.
- Run focused handoff/no-WING/director/guard/rails tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run `git diff --check`, memory lint, and the full pytest suite.

## Risks And Rollback

The main risk is stale fixtures or summary consumers. Rollback would restore the
duplicated literals, but that would reopen a path where operators can use the
wrong expected plan before the read-only WING proof run.
