# ADR: User Presets Physical Readback Contract

## Context

Automixer needs full console read awareness before broad autonomous console
control. The `user_presets` read surface currently exposes cached preset catalog
metadata, but physical enumeration, preset content fingerprints, scoped diff,
and rollback proof are incomplete. Preset recall/store can overwrite console
state, so any promotion path must remain blocked until read-only physical proof
exists.

## Options considered

1. Keep only cached preset DTO coverage.
2. Add direct preset recall/store proof.
3. Add a schema-only physical readback contract.

## Decision

Add a transport-free schema-only physical readback contract for user presets.

## Why this won

The contract names the required future artifact without contacting hardware or
granting authority. It lets no-WING, Product Layer, and frontend surfaces expose
the exact proof gap while preserving safety flags.

## Rejected alternatives

Cached DTO-only coverage was rejected because it cannot validate preset content,
scoped diff, or rollback evidence. Direct preset recall/store proof was rejected
because it is destructive and requires an operator-run physical proof workflow
that is not present yet.

## Implementation plan

- Define `user_presets_physical_readback_contract` and validator.
- Register it as `shadow_only` and non-executing.
- Add it to console read capability tracking.
- Publish fixture/remediation/no-WING summary visibility.
- Normalize Product Layer and frontend claims to read-only, no-live-authority,
  no preset recall/store.

## Test plan

- Contract unit tests for valid artifacts, cached/live rejection, incomplete rows,
  and transport-free source.
- Registry, capability matrix, fixture evidence, remediation, no-WING, Product
  Layer, and frontend adapter tests.
- Full no-WING and backend validation before merge.

## Risks and rollback

Risk: contract visibility may be mistaken for physical proof. Mitigation: keep
`physical_fixture_gap_remains=true`, `accepted_as_physical_fixture_candidate=false`
for contract summaries, and force `preset_recall_allowed=false` plus
`preset_store_allowed=false`.

Rollback: remove the contract module and related registry, no-WING, Product
Layer, frontend, test, and documentation wiring. This returns presets to cached
DTO-only visibility without live authority.
