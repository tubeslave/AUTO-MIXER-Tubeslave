# ADR: WING MVP Fail-Closed Contract Source

## Context

The supervised WING Rack MVP remains blocked on real `WING read-only connection
proof`. Offline/no-WING tooling prepares the operator path, but it must fail
closed consistently before any backend/WING session. The expected-plan
fail-closed conditions were duplicated between the runner, evidence intake, and
release-check validator, and that duplication already drifted once.

## Options Considered

1. Keep duplicated lists and rely on tests.
2. Import one tool module's list into the others.
3. Put the canonical list in `backend/wing_readonly_preflight_contract.py`.

## Decision

Use option 3. The shared contract module now exports
`WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS`.

## Why This Won

The shared contract module already owns WING readonly preflight artifact names,
prestart commands, and safety-boundary constants. It is the right boundary for a
contract that must be shared by runner, evidence, and release-check tooling.

## Rejected Alternatives

Duplicated lists were rejected because they already caused a no-WING gate
mismatch. Tool-to-tool imports were rejected because the contract should not
depend on a specific executable script.

## Implementation Plan

- Add the canonical tuple to `backend/wing_readonly_preflight_contract.py`.
- Use it in `tools/wing_readonly_preflight_runner.py`.
- Alias evidence and release-check public constants to the shared tuple.
- Add tests that assert the aliases use the shared source.

## Test Plan

- Compile touched modules.
- Run focused WING readonly/no-WING tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run the full test suite.

## Risks And Rollback

The main risk is changing list order or losing compatibility names. Rollback
would restore local duplicated tuples, but that would reintroduce the drift risk.
