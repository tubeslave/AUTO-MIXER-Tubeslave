# ADR: WING MVP Handoff Gate Contract Source

## Context

The supervised WING Rack MVP remains blocked on real `WING read-only connection
proof`. Before that proof, the no-WING handoff packet must keep two structured
gates stable: the backend-port prestart gate and the final supervised prestart
gate. Those gates were generated in the handoff builder and independently
reconstructed in the no-WING validator.

## Options Considered

1. Keep duplicated gate literals and rely on tests.
2. Import the handoff builder's expected gate shape into the no-WING validator.
3. Move the gate literals and expected gate builders into
   `backend/wing_readonly_preflight_contract.py`.

## Decision

Use option 3. The shared preflight contract module now owns the handoff gate
step text, indexes, marker lists, runbook target, required decision status, and
expected gate dictionary builders.

## Why This Won

The gate contract is not a tool implementation detail. It is a safety boundary
shared by the handoff packet, no-WING release validator, runbooks, and later
operator review. Keeping it in the shared WING preflight contract avoids
tool-to-tool coupling while preserving fail-closed validation.

## Rejected Alternatives

Duplicated gate literals were rejected because the project has already seen
contract drift in adjacent WING preflight fields. Importing from the handoff
builder into the release validator was rejected because executable tools should
consume a shared contract, not define each other's safety boundary.

## Implementation Plan

- Add shared gate constants and expected gate builders to
  `backend/wing_readonly_preflight_contract.py`.
- Update the handoff packet builder to build both structured gates from the
  shared contract.
- Update the no-WING validator to compute expected gates from the shared
  contract.
- Add focused tests that assert both consumers use the shared constants.

## Test Plan

- Compile touched modules.
- Run focused handoff/no-WING tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run the full test suite.

## Risks And Rollback

The main risk is changing packet shape or operator-step semantics. Rollback
would restore local literals in the builder and validator, but that would
reintroduce drift risk in the first WING read-only operator path.
