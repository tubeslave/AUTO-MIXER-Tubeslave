# ADR — qualify Sec2Drum evidence without promoting model metrics to a production rule

- Date: 2026-10-06
- Status: Accepted
- Update: ML-2026-10-06-1200

## Context

The 09:00 search queued arxiv:2605.13404 at abstract depth. A full review was needed before its reported diffusion and RVQ-CE gains could affect Mixing Learning. The paper is same-lineage evidence, evaluates short windows with automatic metrics and one stochastic sample per condition, and says its cleaned public repository will be released after packaging.

## Decision

Upgrade SRC-ML-20261006-0900-02 to studied_full_text_preprint. Preserve the seconds-aligned interface and short-schedule auxiliary-loss result as bounded research evidence. Do not treat the 72-dimensional PCA subspace, 0/88/164/220 ms context radii, 60 ms window filter, diffusion-step count or lambda 0.10 as mix presets or editing thresholds.

Update existing drum-expression and generated-stem qualification rules instead of creating duplicate rules. Require distributional, paired event/spectral/transient, seed-stability and loudness-matched full-arrangement listening gates. Create EXP-ML-20261006-1200-01 as a one-factor RVQ-CE ablation, but keep it blocked and not_run until public code/checkpoints/manifests plus rights and compute checks are available.

Close the queued static audit ART-ML-20261005-1500-01 at pinned commit c0df19febfb51bc440a30cda4b4bd6a167190d21. It concerns the earlier proof of concept and does not independently reproduce Sec2Drum.

## Consequences

- SRC-ML-20261006-0900-02 is no longer queued and is not independent confirmation of arxiv:2605.10281.
- ART-ML-20261005-1500-01 leaves the queue as a completed static audit; no code was run.
- RULE-ML-20261005-1500-01 and RULE-ML-20261005-1800-02 remain auto_apply:false.
- EXP-ML-20261006-1200-01 remains not_run and blocked.
- The next executable project test remains EXP-ML-20261001-1800-01.
- No audio, DSP, runtime, repository code, model state or paid-compute change is authorized.
