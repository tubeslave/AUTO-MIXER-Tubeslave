# raw-mixer-state-physical-readback-contract-20260605

## Context

`read_surface_remediation_plan` now names `raw_mixer_state_cache` as the next
physical readback target. The action is visible, but the expected artifact shape
is still implicit.

## Goal

Define a transport-free contract for the future raw mixer-state physical
readback artifact so the remediation target is actionable and testable without
contacting hardware.

## Non-goals

- Do not run physical readback.
- Do not query WING or dLive.
- Do not create a live read/write executor.
- Do not close `physical_readback_fixture_missing` from a schema contract alone.

## Safety constraints

- The contract is planning and validation evidence only.
- Cached read DTO fixture coverage remains separate from physical proof.
- Valid artifact validation must not grant live authority or Full Autonomous
  readiness.

## Expected evidence

- A backend contract builder declares required artifact fields and safety flags.
- A validator accepts a minimal physical artifact and rejects cached/live claims.
- Registry, fixture evidence, goal matrix, and no-WING quick checks reference the
  contract as read-only support.
