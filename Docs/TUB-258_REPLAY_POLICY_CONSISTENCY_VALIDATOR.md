# TUB-258 Replay Policy Consistency Validator

## Scope

This validator is replay-only.

- It consumes replay manifests, graph checkpoints, and optional executor traces.
- It does not send OSC.
- It does not modify runtime orchestration.
- It fails closed on dry-run policy mismatches and live-write metadata drift.

## Validation Layers

1. Policy governance
   - manifest must remain `dry_run_only=true`
   - manifest must remain `live_mixer_writes=false`
   - trim snapshot transport policy and timeline policy must agree with the manifest

2. Analyzer to critic coherence
   - every proposal must have an analyzer event
   - analyzer and critic events must preserve proposal `replay_signature`
   - critic event inventory must match proposal critic inventory

3. Critic to decision coherence
   - ranked proposals must map back to graph proposals
   - decision events must preserve rank, signature, and correlation id
   - selected proposal id must match the highest-ranked proposal

4. Decision to executor coherence
   - executor traces must remain dry-run
   - executor events must reference known proposals
   - executor events must preserve replay correlation ids
   - blocked executor events must record a blocked reason

5. Replay restore coherence
   - dry-run-safe manifests are restored through `load_replay_rewind_context()`
   - inconsistent event sequence metadata is rejected as graph drift

## Drift Signals

The validator compares a candidate manifest to a baseline manifest and reports:

- selected proposal drift
- proposal inventory additions/removals
- timeline signature drift
- policy drift in `dry_run_only` and `live_mixer_writes`

## Intended Use

- gate replay fixtures before adding adaptive policy evolution
- detect inconsistent analyzer or critic wiring during refactors
- prove that replay traces remain dry-run-only even when executor simulations are added
- provide a stable compatibility surface for future operator-assisted replay debugging
