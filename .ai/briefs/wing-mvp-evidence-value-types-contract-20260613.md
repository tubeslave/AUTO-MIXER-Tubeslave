# Brief: WING MVP evidence value-types contract

Date: 2026-06-13

## Task

Bind WING read-only preflight evidence intake value types to the shared WING
preflight runner contract.

## Context

The first open MVP blocker is still `WING read-only connection proof`. The real
hardware run must remain bounded to the current read-only proof scope.

The runner and handoff packet already use
`WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES`, but evidence intake used
the broader product live-write policy. If product policy later expands, evidence
could accept a read-only artifact with a value type outside the runner/handoff
contract.

## Scope

- Make evidence intake use the shared WING read-only preflight supported value
  type contract.
- Add no-WING validation that evidence allowed value types match the shared
  contract.
- Keep all behavior offline/read-only.

## Validation

- Focused evidence/no-WING tests.
- no-WING quick gate.
- WING rails tests.
- Full test suite before commit.
