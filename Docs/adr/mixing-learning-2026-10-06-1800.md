# ADR: Metric independence means provenance independence

- status: accepted for candidate evaluation policy
- date: 2026-10-06
- update: `ML-2026-10-06-1800`

## Context

DeePAQ is trained with weak labels derived from ViSQOL MOS and codec bitrate, then evaluated against human listening tests. Its strongest source-separation baseline, the 2f-model, is computed from two PEAQ Basic MOVs. The official 2f supplement reports that scores vary noticeably across PEAQ implementations unless a matching coefficient set is used.

## Decision

Objective endpoints are grouped by provenance rather than product name. Metrics sharing features, pseudo-labels, calibration data, optimized losses or model-selection targets count as one evidence family. Human listener scores remain an external endpoint when they were not used to train or select the reported system.

For ViSQOL, record the pinned implementation/conformance version, audio-versus-speech mode, input sample rate, downmix behavior, reference contract, excerpt activity/duration and treatment-level aggregation. For 2f-model, also record the PEAQ implementation, compatible coefficient set and output clamp.

No metric family may be an automatic mix-quality gate. Acceptance still requires raw diagnostics and randomized loudness-matched listening with protected qualities and cancellation criteria.

## Consequences

- DeePAQ and ViSQOL cannot be counted as two independent votes when ViSQOL supplied DeePAQ training labels.
- DeePAQ's correlations with held-out human ratings remain informative external evaluation evidence.
- ViSQOL cannot cover stereo-image quality because audio mode downmixes to mono.
- 2f results without exact PEAQ/coefficient provenance are not comparable.
- `RULE-ML-20260930-1500-01` and `EXP-ML-20260930-1500-01` are revised; all rules remain `auto_apply:false` and all experiments remain unrun.
