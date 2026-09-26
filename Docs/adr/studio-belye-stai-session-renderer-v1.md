# ADR: Belye Stai Session Renderer Adapter v1

## Context
The routed-compression evaluator now requires an authoritative full-session rerender callback. The delivered `Belye Stai` mix exists as a procedural script with hard-coded project paths and top-level file I/O. The recipe also contains source-dependent routing and shared nonlinear processing, so replacing a dry source in an already-rendered premaster is not exact.

The delivered project has a useful stable boundary: thirteen processed-track prints written after local track EQ/dynamics/level, before group/context routing. That boundary is sufficient to rerun the downstream graph affected by a future insert-style compression hypothesis without pretending that raw-track local processing has already been modularized.

## Options considered
1. Keep the procedural script and spawn it for every candidate.
2. Estimate routed contribution by subtracting a dry source from the premaster.
3. Refactor all raw-track processing and routing in one step.
4. Freeze the processed-track stage as an explicit immutable session input and refactor the downstream graph only.

## Decision
Choose option 4.

Add `audio_workbench/belye_stai_session.py` with a song/version-specific callable `BelyeStaiSessionRenderer`. It accepts the exact thirteen processed-track signals and optional same-shape overrides at that stage, then reruns the downstream graph and returns the existing `SessionRender` contract.

The adapter includes the final bounded drum-accent refinement because the stored premaster used for reproduction includes that operation. It also exposes the frozen deterministic PCM24 dither used by the final delivery so reproduction can be compared at the actual delivered-file boundary.

## Why this won
- It removes hard-coded container paths and top-level project I/O from the rerender path.
- It includes the real source-dependent masking, sends/returns and shared MIX_GLUE stage.
- It is much smaller and safer than migrating every local-track processor at once.
- The override stage is explicit, so future work cannot silently claim a raw-source replacement when only a post-local-processing insert was changed.
- The current delivery can be tested against a concrete immutable reference rather than a proxy metric.

## Rejected alternatives
Direct premaster subtraction remains rejected because source-dependent routing and shared compression make it non-authoritative. Spawning the old script remains path-bound and unnecessarily repeats all local-track DSP. A complete raw-track graph migration is useful later but is too broad for this bounded step and would make reproduction failures harder to localize.

## Implementation plan
- Validate exactly thirteen expected processed tracks, channel layouts, sample rate and frame count.
- Copy and freeze internal track inputs so caller arrays cannot be mutated.
- Rebuild DRUMS and its room send.
- Recompute lead-vocal-driven guitar lift and masking, kick-linked bass ducking, room/chamber/slap returns, shared MIX_GLUE, edge fade/trim and accent refinement.
- Return final mix plus post-shared-processing vocal, drum and early-room anchors.
- Keep Pedalboard Reverb as the frozen default recipe backend but allow a deterministic injected test backend.
- Provide the exact final PCM24 dither/export helper used by the stored delivery.

## Test plan
Synthetic tests must prove:
- empty/no-change and exact-copy override paths are sample-identical;
- vocal override changes the routed full mix and post-shared-processing anchors;
- snare override changes drum bus, room contribution and full mix;
- malformed track sets, unknown overrides and channel-layout mismatches fail closed;
- final PCM24 export is deterministic.

Real-song validation must regenerate the processed-track stage from the original source archives and frozen recipe, render with zero overrides, export with the frozen final dither, and compare to the stored `Belye_Stai_Premaster_44k24.wav`. The acceptance target is exact decoded PCM24 and file SHA-256 identity. Full repository CI must pass on Python 3.10/3.11/3.12 before merge.

## Risks and rollback
The adapter intentionally starts after local track processing. A future compression hypothesis evaluated at this boundary is an insert on the processed track, not a replacement for the raw-track compressor chain. This limitation must remain in metadata and state.

Pedalboard Reverb is part of the frozen real-song recipe. Reproduction evidence is therefore tied to the pinned environment that generated the stored delivery; a future library change may require a new reproduction proof.

Rollback is a normal revert of this isolated module/tests/docs. Existing delivered mixes and source archives are not modified. No audio baseline can be promoted automatically; subjective acceptance remains human-only.
