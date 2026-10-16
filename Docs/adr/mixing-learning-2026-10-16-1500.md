# ADR — 2026-10-16 15:00 MSK: retrieval and tone-generation stay shadow-only

- **Status:** Accepted for research queue
- **Decision ID:** ADR-ML-20261016-1500
- **Related update:** ML-2026-10-16-1500
- **Sources:** arXiv:2610.03405v1; arXiv:2610.08182v1

## Context

PEACE provides a joint code/audio embedding for a bounded Faust library, while CTAG-FX provides a heuristic parameter-role reinterpretation for guitar tone shaping. Both can generate plausible candidates, but neither establishes unique parameter recovery, perceptual benefit in a rock mix, or safe autonomous operation.

PEACE is particularly weak for compressor, gate, limiter and stereo-width retrieval. Its loudness-normalized mid/side magnitude representation removes gain evidence and cannot distinguish mirror panning. CTAG-FX has no listening test, uses one guitar performance, and produced 28 non-finite scrambled renders.

## Decision

1. PEACE-derived results may only rank top-K hypotheses inside a frozen known library, after an effect-activity check.
2. CTAG-FX-style processing may only generate one frozen research candidate chosen before evaluation.
3. Both paths remain shadow-only and `auto_apply:false`.
4. Technical gates precede listening: immutable artifact/manifests, finite/silence/peak checks, raw preservation, stereo/mono checks and reproducible local render.
5. Listening uses separate BS.1770-matched copies while raw copies remain available for measurement.
6. No rule, DSP engine, runtime or model is changed by this update.

## Consequences

The project gains two testable research directions without converting embedding similarity, CLAP alignment or proxy metrics into production truth. The new CTAG-FX A/B remains `not_run_blocked`; the next executable project experiment remains EXP-ML-PENA-AB15-LISTEN-20261010-1800.
