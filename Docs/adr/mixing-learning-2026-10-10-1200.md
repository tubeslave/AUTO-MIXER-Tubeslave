# ADR: Mixing Learning deep review 2026-10-10 12:00 MSK

## Context

The preceding search added a fresh pitch-correction preprint and linked it to the existing BERT-APC source family. A deep comparison was required before any model result, tonal prior or semitone range could influence a project rule.

## Decision

1. Keep `arXiv:2610.11524v1` as a separate dependent-comparator card, not an independent confirmation of BERT-APC.
2. Do not compare the new paper's `83.43%` note accuracy with BERT-APC's original `94.95/89.24% RPA`, or its `3.94` overall mean with BERT-APC's `4.32` pitch MOS. Corpora, frontends, renderers, targets and questions differ.
3. Treat the new paper's BERT-APC row as an independent reimplementation inside the new protocol, not the original end-to-end system.
4. Promote repair, harm, abstention and artistic-intent error as separate evaluation axes. Do not optimise scalar accuracy alone.
5. Keep the vocal-derived tonal prior soft and review-only. Do not adopt `lambda=2.5`, `±1..±3` semitones or a hard key snap as project settings.
6. Tighten the existing pitch rule and replace the older four-way production experiment with a pairwise one-factor A/B. Keep `auto_apply:false` and `not_run_blocked`.
7. Add the pinned BERT-APC demo repository as a demo artifact only; do not treat audio examples, a static page or an MIT license as executable model availability.

## Evidence

- RF-SPC uses a common SOME/RMVPE/NSF-HiFi-GAN path and compares symbolic correction maps on a private paired set.
- Its full hierarchical model improves repair but raises harm relative to flat-head and no-reranking ablations.
- The subjective table reports means without confidence intervals or inferential tests in the visible five-page paper.
- Original BERT-APC uses its own segmentation, stationary-pitch estimator and TD-PSOLA path, different data and different metrics.
- The pinned demo page's MOS values differ slightly from arXiv v3, requiring version pinning.

## Consequences

- No model or rule is promoted to production.
- A valid project A/B must hold timing, gain, renderer, formant mode and residual contour constant, changing only one frozen target/edit map.
- The candidate cannot run until exact vocal material, manual correct/wrong/ambiguous labels and an executable pinned implementation exist.
- Existing drum-bus and subsequent project experiments retain queue priority.

## Status

Accepted as research-base clarification. `citation_check:partial`; no audio, DSP, model or runtime changes.

