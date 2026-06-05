# ADR: dLive State Physical Readback Contract

## Context

Automixer now exposes cached dLive MIDI/TCP state as a read-only Product Layer
DTO, but cached state is not physical proof and is not WING parity proof.
Before dLive state can support promotion decisions for fader, gain, mute,
routing, scene/preset, mix-balance, or performance-adaptation families, the
project needs a schema for separately captured physical readback artifacts.

## Options considered

1. Treat cached dLive state as enough read evidence.
2. Add dLive MIDI/TCP query or apply behavior.
3. Add a schema-only physical readback contract.

## Decision

Use option 3. Add `dlive_state_physical_readback_contract` as a transport-free,
schema-only contract for future operator-run dLive state and WING parity
artifacts.

## Why this won

The contract makes the missing proof concrete without contacting hardware,
sending MIDI/TCP messages, querying dLive, or mutating console state. It lets
no-WING, Product Layer, and frontend read-only surfaces show the exact physical
fixture gap while keeping live authority false.

## Rejected alternatives

Cached-only evidence was rejected because it cannot prove that cached MIDI/TCP
rows match physical console state or WING-oriented parity expectations. Direct
dLive transport query/apply behavior was rejected because it would widen the
runtime surface before readback, rollback, stale-cache, and operator approval
proof exist.

## Implementation plan

- Define `dlive_state_physical_readback_contract` and artifact validator.
- Register the method as `shadow_only`, non-executing, and not part of live
  executor profiles.
- Add dLive state to console read capability, fixture evidence, remediation,
  no-WING summary, Product Layer proof quality, and frontend read-only status.
- Preserve `physical_fixture_gap_remains=true` until a separate operator-run
  hardware artifact is captured and reviewed.

## Test plan

- Contract tests for valid physical artifacts, cached/live rejection, incomplete
  rows, parity fields, and transport-free source.
- Registry, project capability, fixture evidence, remediation, no-WING, Product
  Layer, and frontend adapter tests.
- Full backend suite, no-WING quick validation, memory lint, and diff hygiene.

## Risks and rollback

Risk: a passing schema gate could be mistaken for real dLive or WING hardware
proof. Mitigation: all summaries keep `live_write=false`,
`live_authority_granted=false`, `midi_send_allowed=false`,
`tcp_send_allowed=false`, `dlive_write_allowed=false`,
`dlive_apply_allowed=false`, and `physical_fixture_gap_remains=true`.

Rollback: remove the contract module plus registry, capability, fixture,
remediation, no-WING, Product Layer, frontend, test, and ADR wiring. The dLive
surface returns to cached DTO visibility only, without live authority.
