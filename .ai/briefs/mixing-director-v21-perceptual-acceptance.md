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

Focused reproduction of the committed perceptual-critic module/tests: `6 passed`.

The first full matrix run, workflow `35835155423`, reached an existing Artifact Critic test before the new perceptual tests and exposed a real low-sample-rate bug: the 6-14 kHz fizz band is invalid at an 8 kHz sample rate, and the click-regression tolerance was dimensionally too large for a per-sample click rate. The failure was not in the new perceptual gate.

Revision commit `1b6286a1698223418024a3316816cddba6a6e3e4` makes Artifact Critic band filtering Nyquist-safe and uses a click-rate-specific regression tolerance. The exact failing synthetic fixture was reproduced after the fix: clean click rate `0.0`, injected-click rate `0.0049375`, and the comparison rejects the candidate for `click_rate` regression. Full CI rerun `35835666282` is pending at the time of this note.

## Next integration

Autonomous Iteration v2 should combine this perceptual acceptance result with the existing electrical/tonal guards and accept a candidate only when both layers pass. Subjective sonic acceptance remains human-in-the-loop during development.
