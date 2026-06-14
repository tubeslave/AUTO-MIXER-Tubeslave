# ADR: WING MVP Prestart Backend No-Authority Detail

## Context

The supervised WING MVP cannot move beyond the read-only connection proof until
a real WING read-only artifact is captured and reviewed. Offline tooling can
prepare that run, but the real preflight runner must fail closed before WebSocket
if the no-WING prestart summary is stale, minimal, or does not explicitly prove
that the backend-port gate granted no live authority.

## Options considered

1. Accept a passed `backend_port_free` status without explicit detail flags.
2. Require explicit backend-port no-authority detail fields in the no-WING
   summary and expected plan.

## Decision

Use option 2.

## Why this won

Explicit `false` fields are stronger evidence than omission and make stale
prestart summaries fail closed before any backend WebSocket connection. This
keeps the first WING blocker honest: offline checks prepare a real read-only
proof but never grant live authority or close readiness.

## Rejected alternatives

- Status-only validation was rejected because it cannot distinguish current
  no-authority prestart evidence from an older minimal summary.
- Documentation-only validation was rejected because the runner itself owns the
  pre-WebSocket safety boundary.

## Implementation plan

- Define the required false backend-port detail fields in the shared read-only
  preflight contract.
- Emit those fields from the no-WING backend-port check.
- Validate those fields in the read-only preflight runner and include them in
  expected-plan matching.
- Preserve frontend E2E generated `.last-run.json` during no-WING command
  checks.

## Test plan

- Focused runner and backend-port pytest coverage.
- no-WING quick `--skip-frontend` in the restricted sandbox.
- `git diff --check`.

## Risks and rollback

The change may reject older prestart summaries. Roll back by removing the
required false-field validation, but only if operators cannot rerun the current
no-WING gate before WING work.
