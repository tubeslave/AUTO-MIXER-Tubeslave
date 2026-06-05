# Brief: dLive State Physical Readback Contract

## Task

Add a schema-only physical readback contract for `dlive_state` so cached
dLive MIDI/TCP state cannot be mistaken for physical console proof or WING
parity evidence.

## Context

`build_dlive_state_snapshot` exposes cached dLive names, fader raw values, mute
state, raw MIDI cache counts, freshness, and parity blockers without querying or
mutating hardware. The project still lacks a physical readback artifact schema
for proving that cached dLive state matches console state and maps safely to
WING-oriented promotion surfaces.

## Safety Constraints

- Do not add MIDI/TCP sends, dLive writes, scene recalls, or console queries.
- Keep the method `shadow_only` and outside live executor profiles.
- Keep `physical_fixture_gap_remains=true` and no live authority.
- Normalize source claims for `midi_send_allowed`, `tcp_send_allowed`, and
  `dlive_write_allowed` to false in Product Layer/UI.

## Deliverables

- Backend contract builder and validator.
- Registry, capability matrix, fixture evidence, remediation, no-WING summary,
  Product Layer, frontend, and tests.
- ADR and durable memory update.

## Validation

Run focused backend tests, frontend adapter tests, no-WING quick validation,
memory lint, frontend build, and full backend pytest before commit.
