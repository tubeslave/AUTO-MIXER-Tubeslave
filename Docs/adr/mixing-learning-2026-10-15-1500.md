# ADR: FX tool plans remain shadow-only until they beat bypass

- **Date:** 2026-10-15
- **Update:** `ML-2026-10-15-1500`
- **Status:** accepted for research queue; no automatic application

## Context

LLM2Fx-Tools reports better effect selection/order and listening results than several baselines for synthetic, single-instrument FX-chain recovery. The inverse mapping is non-unique, the dataset excludes repeated effect types, language/CoT supervision is model-generated, and the public repository currently lacks a released LP-Fx dataset, a pinned LLM2Fx-Tools checkpoint manifest and an explicit repository licence file.

A valid tool call therefore proves only that a schema was executed. It does not prove sonic benefit, causal correctness, full-mix transfer or legal/reproducible deployability.

## Decision

1. Treat any LLM-generated FX chain as a versioned shadow proposal, never as an automatic rule.
2. Require a bypass arm and a hard harm gate in every first audio test.
3. Freeze the complete chain JSON, renderer/backend/version, sample/channel contract and artifact hashes before rendering.
4. Preserve raw measurement renders; create separate BS.1770-matched, true-peak-safe copies for blind listening.
5. Use one isolated project-owned stem first because the paper does not validate full mixes.
6. If the chain wins, identify causality only in later one-effect removal experiments.
7. Do not transfer Appendix G sampling ranges as rock presets.
8. Keep `auto_apply:false`; the initial test is `not_run_blocked` until dataset/checkpoint/licence manifests are available.

## Consequences

- Tool-call auditability can be researched without changing DSP or production code.
- Bypass remains an explicit contender and prevents a candidate from passing merely because it processes audio.
- Explanations are logged as hypotheses and checked against measured signal changes.
- Serial compression/EQ and full-mix use remain outside the evidence boundary.
