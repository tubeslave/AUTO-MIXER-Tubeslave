# STUDIO Mastering Meter Safety — 2026-09-23

## Goal

Create one trustworthy objective mastering-safety primitive for the offline/STUDIO pipeline before adding more autonomous mastering control.

## Repository evidence

`CLAUDE.md` and `.cursorrules` require integrated LUFS with ITU-R BS.1770-4 double gating and true peak with 4x oversampling. `backend/auto_mastering.py` still uses an RMS loudness approximation and a sample-peak ceiling in its built-in path. The repository already contains `KWeightingFilter` and `TruePeakMeter` in `backend/lufs_gain_staging.py`.

## Bounded task

Add a side-effect-free STUDIO mastering meter that:

- reuses the existing K-weighting and 4x true-peak implementation;
- computes gated integrated loudness on 400 ms blocks with 100 ms hop;
- reports sample peak, true peak, RMS and crest factor;
- provides a safety-only true-peak attenuation helper that never adds gain;
- has synthetic regression tests, including an intersample-peak case where sample peak is safe but dBTP is not.

Do not change creative mastering output or promote any audio baseline in this task.

## Acceptance

- Synthetic 1 kHz full-scale mono sine measures about -3.05 LUFS.
- Identical dual-mono stereo measures about +3.01 LU relative to mono.
- Intersample overshoot is detected above sample peak.
- True-peak safety attenuation brings the candidate to the requested ceiling and never boosts already-safe audio.
- Full repository CI must pass before merge.

## Human listening

Not required for accepting this measurement-only primitive. Human listening remains mandatory when the meter is wired into any mastering change that alters audio.
