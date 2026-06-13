# ADR: WING MVP Handoff Packet Output Contract

## Context

The current MVP remains blocked on real `WING read-only connection proof`.
Before any supervised proof, the operator must use the handoff packet to run the
no-WING prestart gate and the read-only WING preflight.

The review-template and review-report paths are contract-owned, but the handoff
packet output path was still duplicated in handoff/no-WING tooling.

## Options Considered

1. Keep the handoff packet output path duplicated.
2. Move the path into the shared WING preflight contract and validate no-WING
   handoff command output.
3. Only change the handoff builder default.

## Decision

Use option 2. The shared preflight contract now owns
`WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT`, and no-WING evidence validation
fails closed when the handoff command `--output` path drifts.

## Why This Won

The handoff packet is the operator-visible entry point for the proof chain. Its
output path must be stable and machine-checkable before a real WING session.

## Rejected Alternatives

Duplicated literals were rejected because they allow silent drift. Builder-only
centralization was rejected because no-WING still needs to reject stale command
values.

## Implementation Plan

- Add the handoff packet output constant to the shared contract.
- Make handoff builder and no-WING consume the shared constant.
- Add no-WING detail/summary fields for the parsed output and contract match.
- Fail closed on `wing_readonly_preflight_handoff_packet_output_mismatch`.
- Cover the happy path and stale path in focused tests.

## Test Plan

- Compile touched modules.
- Run focused handoff/no-WING/director/guard/rails tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run `git diff --check`, memory lint, and the full pytest suite.

## Risks And Rollback

The main risk is stale fixtures or summary consumers. Rollback would restore the
duplicated literals, but that would reopen a path where operators can be pointed
at the wrong handoff packet.
