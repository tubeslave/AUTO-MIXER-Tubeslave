# bus-group-main-dca-physical-readback-contract-20260605

## Context

`bus_group_main_dca_state` is a blocking read surface for grouped balance,
DCA, main-bus optimization, and performance-adaptation promotion targets. The
repo already exposes cached bus/main/DCA DTO state, but cached state is not
physical proof that the actual console matches the Product Layer view.

## Goal

Add a transport-free bus/group/main/DCA physical readback contract, wire it into
registry/read-surface/no-WING/Product Layer/frontend summaries, and keep the
physical fixture gap explicit.

## Non-goals

- Do not query a console.
- Do not run physical readback.
- Do not accept cached mixer state as physical proof.
- Do not grant bus, group, DCA, main-bus, matrix, mute, or Full Autonomous live
  authority.

## Safety constraints

- Contract readiness is schema evidence only.
- Product Layer and frontend must normalize any live-authority or
  full-autonomous claims to false.
- `physical_fixture_gap_remains` must stay true until a separate operator-run
  artifact is captured and reviewed.

## Expected evidence

- Registry exposes `bus_group_main_dca_physical_readback_contract` as
  `shadow_only`.
- Fixture/remediation summaries name the contract while keeping
  `physical_readback_fixture_missing`.
- no-WING release summary includes a bus/group/main/DCA contract schema gate.
- Product Layer and cockpit module status show the contract as read-only
  planning evidence.
