# STUDIO Mastering Meter Integration — 2026-09-23

## Goal

Wire the already accepted `StudioMasteringMeter` into the existing offline `AutoMaster` processing paths without breaking legacy callers or changing the public return shapes.

## Repository evidence

`backend/auto_mastering.py` still uses RMS as a loudness proxy in `_builtin_master()` and `_match_target_loudness()`, and its built-in limiter is sample-peak based. PR #112 introduced a tested BS.1770-style integrated loudness meter plus 4x true-peak safety attenuation in `backend/studio_mastering_metrics.py`.

## Bounded task

- Keep `_estimate_lufs()` and `_limit()` as compatibility helpers for existing tests/callers.
- Use `StudioMasteringMeter.integrated_lufs()` for actual built-in and fallback loudness targeting.
- Use `StudioMasteringMeter.limit_true_peak()` in actual built-in/fallback/reference mastering output paths.
- Report integrated LUFS from actual `MasteringResult` objects instead of RMS mislabeled as LUFS.
- Add focused regression tests for the real paths, including an intersample-peak case.

Do not change mix balance, EQ matching policy, compressor policy, or any user audio baseline in this task.

## Acceptance

- Existing AutoMaster API tests still pass.
- Built-in mastering result LUFS matches `StudioMasteringMeter` on the rendered result.
- Actual loudness-target path catches a case where sample peak is below -1 dBFS but 4x true peak is above -1 dBTP.
- Full repository CI passes before merge.

## Human listening

The code can alter rendered mastering gain/ceiling behavior, so objective CI acceptance is not subjective audio acceptance. Any audio produced by the changed path remains pending human A/B before it may become a project baseline.