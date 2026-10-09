# ADR — PEACE requires a pinned, topology-fixed shadow protocol

- **Date:** 2026-10-09
- **Update:** `ML-2026-10-09-1200`
- **Status:** accepted for research queue; no production implementation

## Context

The morning search retained PEACE as an offline retrieval candidate. Deep review of the released inference code, audio frontend, model-loading path, notices and tests found useful guards but no released-checkpoint metric reproduction. It also exposed a provenance gap: the paper describes 10-second rendered pairs, while the released README says training used 6-second excerpts, without documenting the crop rule.

The morning experiment also compared parameter-visible exact-program retrieval with masked topology retrieval. Those are different tasks and cannot support a clean A/B conclusion.

## Decision

1. Keep PEACE in shadow/offline retrieval only; `auto_apply:false`.
2. Pin `DBraun/PEACE@2d8a3887501bcdb4779544295845b1e69b6b9b78` and, before execution, pin the Hugging Face commit plus hashes of config, weights and tokenizer.
3. Freeze the full preprocessing and crop contract: 48 kHz, 6.0-second model input, -18 LUFS normalization, peak cap, zero-padded log-mel frontend, crop offset and source-render duration.
4. Evaluate topology with one canonical gallery item per topology. Compare parameter-visible and parameter-masked embeddings while holding query audio, target, gallery and all preprocessing fixed.
5. Report per-family/source/chain-length results and inactive-effect negatives; do not transfer aggregate or reverb results to dynamics/nonlinear processing.
6. Treat MIT code, CC BY-NC weights and renderer-source licences as separate gates.
7. Do not interpret retrieval as musical quality, parameter estimation, absolute loudness, true peak, L/R direction or phase.
8. Do not displace the practical queue: `EXP-ML-20261001-1800-01` remains the next executable loudness-matched musical A/B.

## Consequences

`EXP-ML-20261009-0900-01` remains `not_run_blocked`, but now has one target and one changed factor. A later musical A/B requires a normal audio-processing variable, raw and BS.1770-matched renders, protected-quality checks and Dmitry's blinded evaluation. No audio, code, DSP, runtime or model state changed.
