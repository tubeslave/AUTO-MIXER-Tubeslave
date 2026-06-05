# FX Slots/Inserts Physical Readback Artifact Intake Brief

## Task

Add a read-only artifact intake for operator-captured FX slots/inserts physical
readback JSON.

## Scope

- Accept explicit, inline, or discovered JSON artifacts for
  `fx_slots_and_inserts`.
- Validate artifacts against the existing FX slots/inserts physical readback
  contract.
- Surface valid artifacts only as manual-review fixture candidates.
- Wire registry, capability matrix, no-WING summary, Product Layer, frontend
  evidence, and tests.

## Safety Constraints

- No console query, transport, write, FX parameter mutation, insert assignment
  mutation, or test execution authority.
- Keep `fixture_closure_allowed_by_intake=false`.
- Keep `physical_fixture_gap_remains=true`.
- Keep `live_authority_granted=false`.
- Keep `full_autonomous_ready=false`.

## Done Criteria

- Missing artifacts fail closed as operator-required evidence.
- Valid artifacts are visible as review candidates only.
- Invalid artifacts block the release gate.
- no-WING, Product Layer, and frontend normalize unsafe claims to no authority.
