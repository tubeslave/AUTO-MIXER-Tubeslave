# ADR-ML-20261016-1800 — Require version, licence, baseline and renderer gates before PEACE/CTAG-FX experiments

- Status: accepted_for_research_workflow
- Date: 2026-10-16
- Update: [ML-2026-10-16-1800](../mixing_learning_updates/ML-2026-10-16-1800.md)
- Scope: research and experiment planning only
- Runtime effect: none

## Context

PEACE and CTAG-FX expose promising representations and configuration mechanisms, but the currently available evidence cannot justify automatic mix decisions.

PEACE has a public repository and released weights, yet the source and model artifacts have different licence and provenance requirements. Its reported retrieval is evaluated inside a fixed synthetic Faust library, and a supervised closed-set classifier outperforms the retrieval model on the reported ordered-pair benchmark. The arXiv and camera-ready numerical paths also differ because of feature padding.

CTAG-FX exposes an official demo and at least one normalized configuration, but the renderer implementation, mapping, stabilizer, dependency lock and licence are not yet public. The demo “RAW” guitar is already processed, and its variants are not loudness matched.

## Decision

1. Treat arXiv:2610.03405v1 as the canonical PEACE numerical record where it differs from the camera-ready paper. Count both as one evidence family.
2. Keep PEACE research-only and shadow-only within its known 21-effect serial library.
3. Require an immutable source commit, immutable model revision, file hashes, dependency lock, renderer parity, fixture manifest and rights decision before execution.
4. Compare PEACE against a simple closed-set classifier and documented heuristic. Do not presume retrieval superiority.
5. Require an explicit uncertainty/none output; a missing top-K candidate must never be treated as proof that an effect or chain is absent.
6. Treat PEACE code, weights and effect sources as a composite licence surface. Top-level MIT does not authorize incompatible weight or compiled-effect use.
7. Keep CTAG-FX blocked until native executable code, processor mapping, stabilizer, dependencies and licence are frozen.
8. Do not use official demo audio as a controlled bypass comparison.
9. When CTAG-FX becomes executable, compare native bypass with exactly one frozen native configuration on truly dry project-owned guitar. Preserve raw renders and produce separate attenuation-only BS.1770 loudness-matched listening copies.
10. Keep NAM parity separate from the creative processor A/B.
11. All candidate rules remain auto_apply:false and all unexecuted experiments remain not_run or not_run_blocked.

## Consequences

- PEACE may be evaluated only after its closed-set baseline and artifact gates are satisfied.
- CTAG-FX may be studied from text and demo metadata, but not promoted from demo listening impressions.
- Model or renderer execution cannot begin merely because source code is available.
- Reproducibility and rights failures are cancellation conditions, not documentation footnotes.
- No DSP engine, runtime rule, audio or model is changed by this decision.
