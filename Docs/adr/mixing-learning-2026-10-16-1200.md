# ADR: ML-2026-10-16-1200

## Context

The preceding search queued a deep audit of two fresh preprints: an Opus transparency listening study with released tables, and `vst-bench-ml` with a public package/companion host. The project needs bounded evidence for listening design and plugin-render provenance without promoting either result into DSP defaults.

## Decision

1. Treat the Opus paper's “160 kbps” as a configured unconstrained-VBR target, not a fixed observed bitrate or project delivery threshold. The released Phase-2 sample averages span 159–213 kbps.
2. Describe Phase 2 as a BLUP-selected critical-listener/material stress set, not an independently recruited expert validation.
3. Mark reproducibility partial: released CSVs support descriptive recounts, but analysis code, stimuli/encoded files and trial/session/order fields are absent.
4. Require explicit ordering/code for overlapping screening filters. The retained set is reproducible as a union, while the reported protocol-versus-anchor attribution is ambiguous.
5. Separate core `vst-bench-ml` manifest integrity from complete host-side kept-attempt evidence.
6. For plugin-rendered research, retain immutable raw pre-normalization output, build matched derivatives separately, and run per-channel plus M/S QC. The default normalized-only export is insufficient as sole evidence for gain, dynamics, clipping or stereo behavior.
7. Do not generalize the validated macOS-arm64 dual-mono synthetic fixture to third-party stereo effects or other platforms.
8. Add a local Opus off/on experiment only for a real delivery decision. Preserve the existing PENA listening test as the next executable priority.

## Consequences

- `RULE-ML-20261016-0900-01` and `EXP-ML-20261016-0900-01` advance to revision 2 with raw-preservation and stereo-QC gates.
- `RULE-ML-20261016-1200-01` prevents codec-target transfer without actual-bitrate and project-listening evidence.
- `EXP-ML-20261016-1200-01` remains `not_run` and `auto_apply:false`.
- No audio was downloaded or rendered, no source code was executed, and no DSP/runtime/model or production code changed.
