# ADR: Mixing Learning deep review 2026-10-05 18:00 MSK

- decision ID: ML-2026-10-05-1800
- status: accepted as knowledge-base update; no automatic production action
- mode: deep_review
- preceding update: ML-2026-10-05-1500
- preceding successful search: ML-2026-10-05-1500

## Decision

Upgrade the P-center and Separate-and-Detect cards from abstract/metadata depth to full-text status. Narrow the prior P-center wording: bimodal click placement does not establish a universal two-event boundary, and 20/40/80 ms research conditions are not edit thresholds.

Treat Separate-and-Detect as a benchmarked research pipeline with class-specific kick/snare advantages, not as an overall win over direct ADT and not as listener-validated editable stems. Retain its open repository and released MIT-tagged checkpoints as an inspectable but not independently reproduced artifact.

Adopt two bounded candidate rules: perform a perceptual timing audit before cross-instrument correction, and qualify generated drum stems on separate event-accuracy, leakage, bandwidth, reconstruction and preference axes.

## Evidence boundary

The P-center evidence comes from two small laboratory experiments with trained musicians, fixed compound sounds and hand-matched relative loudness. It has no full-mix preference test.

Separate-and-Detect uses synthetic-heavy training, mono 16 kHz audio and objective event/separation metrics. The direct ADTOF comparator remains stronger in overall F1 on MDB and ENST, while the proposed system is stronger only for kick and snare. No formal listening test or independent reproduction was found.

The code repository and Hugging Face metadata report MIT licensing, but dataset and audio rights remain separate. No checkpoint, dataset or audio was downloaded, and no GPU work was run.

## Consequences

- RULE-ML-20261005-1800-01 and RULE-ML-20261005-1800-02 remain auto_apply:false.
- EXP-ML-20261005-1500-01 is narrowed to event-specific P-center-aware timing and remains not_run.
- EXP-ML-20261005-1800-01 is not_run and blocked by rights, compute and reproduction audit.
- The next executable project test remains EXP-ML-20261001-1800-01.
- No audio, DSP, runtime, repository code or model state changes.
