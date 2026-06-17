# WING Readonly Runner Verifier Safety Flag Gate

## Context

The WING read-only preflight runner validates the detached no-WING artifact
manifest verifier before any backend/WING WebSocket opens. The verifier report
must preserve the offline safety boundary: read-only, no live writes, no backend
or WING contact, no console mutation, no console queries, and no test execution.

## Options considered

1. Keep only generic verifier report value mismatch diagnostics.
2. Add dedicated violations for top-level verifier safety flag mismatches.

## Decision

Add dedicated pre-WebSocket violations when the verifier report mismatches these
safety fields:

- `read_only`
- `live_write`
- `contacts_wing`
- `contacts_backend`
- `mutates_console_state`
- `queries_console`
- `executes_tests`

## Why this won

Safety-boundary mismatches should be operator-visible without requiring someone
to inspect a generic mismatch list. The runner already fails closed; explicit
violations make the repair target concrete before transport opens.

## Rejected alternatives

Relying on generic report value mismatch was rejected because it treats
safety-boundary defects like ordinary artifact consistency mismatches.

## Implementation plan

- Extend `tools/wing_readonly_preflight_runner.py` with dedicated safety flag
  mismatch violations.
- Add a pre-WebSocket regression test that mocks verifier safety flag claims.

## Test plan

- Focused runner verifier tests.
- Focused WING/no-WING tests.
- Standard full pytest command.

## Risks and rollback

Risk is limited to stricter offline/pre-WebSocket diagnostics. Rollback is to
remove the dedicated violations while preserving the generic verifier failure.
