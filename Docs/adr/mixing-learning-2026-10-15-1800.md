# ADR: LLM2Fx listening claims remain exploratory and artifact-gated

- **Date:** 2026-10-15
- **Update:** `ML-2026-10-15-1800`
- **Status:** accepted for research queue; no automatic application

## Context

`ML-2026-10-15-1500` admitted LLM2Fx-Tools only as a shadow FX-chain planner with a bypass harm gate. Deep review compared its published listening method with ITU-R BS.1534-3 and audited the pinned public code tree.

The paper reports 8 questions, 16 participants, hidden-reference mean 81.5 and pairwise t-tests, but the read text does not document listener screening, anchors, listening conditions, median/IQR, confidence intervals, a repeated-measures omnibus model or multiplicity control. The public repository contains useful model/training/preprocessing code, but no example training config, checkpoint, dataset release, inference/evaluation/listening scripts, frozen environment or stimuli/results manifest; a dataset import path also does not match the pinned tree.

## Decision

1. Treat reported listening superiority as exploratory, not project-ready confirmation.
2. Keep `RULE-ML-20261015-1500-01` shadow-only and lower confidence to `low_research_only`.
3. Keep `EXP-ML-20261015-1500-01` blocked until the complete paper-parity manifest exists.
4. Require renderer nulls before creative A/B, with bypass as A and exactly one frozen chain as B.
5. For repeated listener/material observations, use listener/excerpt-aware repeated-measures or preregistered paired-resampling analysis; never pool observations into independent t-tests.
6. Treat Fx-Removal/Fx-Norm and related encoder metrics as dependent parts of the system that require separate controls.

## Consequences

- No numerical FX settings are adopted.
- No audio, model, DSP or runtime is changed.
- A future executable test needs manifest, licence and null-test clearance before listening.
- Acceptance still depends on loudness-matched blind preference by Дмитрий and preserved attack, timbre, stereo/mono, noise/bleed, tails, role and intent.

## Supersession

This ADR refines, but does not revoke, `mixing-learning-2026-10-15-1500.md`.
