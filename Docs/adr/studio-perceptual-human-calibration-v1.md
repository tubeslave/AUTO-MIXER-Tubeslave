# ADR: STUDIO Perceptual Human Calibration v1

## Decision

Keep the existing Perceptual Critic acceptance thresholds unchanged. Add a separate calibration-evidence layer that records disagreements between bounded machine gates and human level-matched listening without granting either layer authority it does not have.

A positive human preference may demonstrate that a target proxy missed an audible property. It does **not** erase protected regressions, retroactively change a failed target into a pass, or promote an audio baseline. A machine-safe result likewise does not override a negative human listening verdict.

## Motivating evidence

The Belye Stai v2 `balanced` vocal candidate was rejected by the existing `vocal_intelligibility` target: `0.6800658645 -> 0.6714029435`, delta `-0.0086629210`, with failure `target_not_improved`. The same critic result had no protected regressions. The user later accepted that named vocal comparison because the vocal was more stable in the mix, did not jump out or disappear, and felt correctly placed.

This is recorded as `target_proxy_miss_candidate`, not as a retroactive critic pass. The metric hypothesis is foreground/phrase stability, which is a different property from the existing 1.2-4.5 kHz energy-ratio intelligibility proxy.

## New diagnostic metric

`foreground_stability_evidence()` measures robust P90-P10 RMS spread and P95 absolute level deviation over fixed activity windows derived from the baseline (or an explicit fixed reference). A candidate receives one constant median-level match before comparison, so a simple fader gain cannot masquerade as improved stability.

The metric is diagnostic only in v1:

- it has no production acceptance threshold;
- `accept` is `None`;
- it cannot promote a baseline;
- human listening remains required;
- the existing Perceptual Critic and its thresholds are untouched.

A deterministic synthetic calibration case reduced phrase-level spread from `8.3182 dB` to `1.2966 dB` while the candidate was deliberately made about 5 dB louder. Constant level matching removed that loudness advantage and the stability improvement remained `7.0216 dB`, proving the diagnostic is not merely a loudness detector.

## Disagreement classifications

The calibration layer distinguishes:

- target-only machine rejection + human acceptance -> `target_proxy_miss_candidate`;
- protected machine regression + human acceptance -> `human_preference_safety_conflict`, safety rejection remains intact;
- machine-safe + human rejection -> `machine_false_positive_candidate`;
- positive/negative agreements -> record evidence without autonomous taste authority.

## Boundaries

No live DSP, mastering, accepted Belye Stai vocal settings, compressor core, Autonomous Iteration promotion policy, or existing Perceptual Critic thresholds are changed. No neural audio or paid external credits are used. Real-song foreground-stability scoring still requires the corresponding isolated baseline/candidate foreground stems; those bytes are not available in the current runtime, so no real-song stability number is invented.
