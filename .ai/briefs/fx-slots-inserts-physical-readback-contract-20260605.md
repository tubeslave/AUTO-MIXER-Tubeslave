# FX Slots/Inserts Physical Readback Contract

## Task

Define a schema-only physical readback contract for `fx_slots_and_inserts`.

## Context

`fx_slots_and_inserts` already exposes cached Product Layer DTOs for FX slot
models, bypass/mix/source assignment, parameter schema hints, and insert
assignment state. That cache evidence is useful for operator visibility, but it
is not physical proof and must not unlock FX, insert, routing, or mix-balance
promotion.

## Safety Constraints

- Do not query or mutate a live console.
- Keep FX and insert assignment apply blocked/manual-only.
- Keep `physical_fixture_gap_remains=true`.
- Keep `live_authority_granted=false` and `full_autonomous_ready=false`.

## Expected Output

- Transport-free backend contract and validator.
- Registry, remediation, no-WING, Product Layer, and frontend visibility.
- Tests proving schema-only behavior and safety normalization.
