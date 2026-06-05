# Routing/Sends Physical Readback Contract - 2026-06-05

## Context

`routing_and_sends` has cached/static Product Layer coverage for Dante roles,
channel bus/main/matrix sends, and output route mappings, but it still lacks a
physical readback artifact contract. Routing, FX-send/insert, scene/preset, and
mix-balance promotions must not treat cached routing/send state as hardware
proof.

## Task

Add a read-only, transport-free contract that describes the future physical
routing/sends readback artifact and exposes it through no-WING, Product Layer,
registry, frontend status, and project-goal capability metadata.

## Constraints

- No console queries or live writes.
- Keep `physical_fixture_gap_remains=true`.
- Normalize any source live-authority or full-autonomous claims to false.
- Preserve existing routing/sends cache fixtures as partial evidence only.
- Validate through focused backend/frontend tests and no-WING release checks.

## Expected Outcome

Operators can see that routing/sends has a schema contract for future physical
proof, while all promotion and autonomous gates remain blocked until a real
operator-run hardware readback artifact exists.
