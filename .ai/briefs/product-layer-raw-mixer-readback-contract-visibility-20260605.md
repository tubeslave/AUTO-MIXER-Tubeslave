# product-layer-raw-mixer-readback-contract-visibility-20260605

## Context

no-WING release summaries now validate
`raw_mixer_state_physical_readback_contract`, but Product Layer proof evidence
only normalizes read-surface fixture quality. Operators can see the next readback
target and remediation action, but not whether the raw mixer-state artifact
contract itself passed the release gate.

## Goal

Expose raw mixer-state physical readback contract status through the read-only
Product Layer and cockpit Module Status surfaces.

## Non-goals

- Do not run physical readback.
- Do not validate a real hardware artifact from Product Layer.
- Do not mark `physical_readback_fixture_missing` as closed.
- Do not add live read/write authority.

## Safety constraints

- Normalize source safety flags to `read_only=true`, `live_write=false`, and
  `full_autonomous_ready=false`.
- Keep `physical_fixture_gap_remains=true` visible.
- Treat contract target mismatch as a review warning, not a live action.

## Expected evidence

- `build_supervised_proof_evidence_snapshot()` includes
  `raw_mixer_state_physical_readback_contract`.
- Frontend Module Status shows contract status, target consistency, and
  no-live-authority tags.
- Settings proof inspector shows contract availability/gap/authority state.
