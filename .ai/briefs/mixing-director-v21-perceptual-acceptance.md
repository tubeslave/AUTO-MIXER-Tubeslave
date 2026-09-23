# Mixing Director v2.1 — Perceptual candidate acceptance

## Goal

Turn the v2.0 Perceptual Mix Critic from a diagnostics-only observer into a bounded regression gate for autonomous studio iterations.

## Change

`audio_workbench/mixing/perceptual_critic.py` now exposes `accept_candidate(before, after, target)` plus `PerceptualAcceptancePolicy`.

The gate is deliberately local to one hypothesis. It does not assign an overall mix-quality score and it does not replace human listening.

Supported targets:
- vocal intelligibility: must increase;
- harshness: must decrease;
- drum punch: must increase;
- climax lift: must increase.

A candidate is rejected when the intended target does not clear its minimum improvement or when it causes excessive collateral drift in density, stereo width, foreground level, harshness, intelligibility, punch, or climax lift.

Peak headroom, integrated loudness and broad tonal-shift guards remain in the outer Autonomous Iteration layer, because the perceptual snapshot does not own those measurements.

## Threshold status

Current thresholds are conservative engineering defaults, not psychoacoustically validated constants. They must be calibrated against level-matched human A/B evidence before being treated as production policy.

## Verification

Focused local test reproduction of the committed module/tests: `6 passed`.

GitHub CI for commit `dad336fbdc5b8d81b254cde57ee57e763ddea7c5` was started as workflow `35835155423`; final matrix status is recorded separately in Director state when available.

## Next integration

Autonomous Iteration v2 should combine this perceptual acceptance result with the existing electrical/tonal guards and accept a candidate only when both layers pass. Subjective sonic acceptance remains human-in-the-loop during development.
