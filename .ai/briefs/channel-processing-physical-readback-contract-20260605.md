# channel-processing-physical-readback-contract-20260605

## Context

`channel_eq_dynamics_gate` is a blocking read surface for EQ, compressor, gate,
de-esser, and cross-channel unmasking promotion targets. Current cached DTO
fixtures can describe channel processing state, but the repo still needs a
physical readback artifact contract before any operator-run WING/dLive readback
can be accepted as proof.

## Goal

Add a transport-free channel processing physical readback contract, wire it into
registry/read-surface/no-WING/Product Layer summaries, and keep the physical
fixture gap explicit.

## Non-goals

- Do not query a console.
- Do not run physical readback.
- Do not accept cached mixer state as physical proof.
- Do not grant EQ, compressor, gate, de-esser, unmasking, or Full Autonomous
  live authority.

## Safety constraints

- Contract readiness is schema evidence only.
- Product Layer and frontend must normalize any live-authority or
  full-autonomous claims to false.
- `physical_fixture_gap_remains` must stay true until a separate operator-run
  artifact is captured and reviewed.

## Expected evidence

- Registry exposes `channel_processing_physical_readback_contract` as
  `shadow_only`.
- Fixture/remediation summaries name the contract while keeping
  `physical_readback_fixture_missing`.
- no-WING release summary includes a channel processing contract schema gate.
- Product Layer and cockpit module status show the contract as read-only
  planning evidence.
