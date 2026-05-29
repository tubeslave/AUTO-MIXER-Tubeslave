Gain Staging Research Report

Hypothesis

Before real WING testing, the method must say whether the measurement context itself is good enough. A correct-looking hold/up/down decision can be misleading if the capture lacks spectral bands, noise-floor reference, or enough active samples.

Changes

Added `measurement_quality` to `foh_summary`:

- readiness classes: `full_context`, `usable_without_noise_floor`, `limited_spectral_context`, `level_only`, `short_window`, and `insufficient_signal`;
- warnings for short analysis windows, insufficient signal presence, missing spectral fingerprint, limited spectral band count, missing noise-floor reference, and few active samples;
- sample counts, active sample counts, signal presence ratio, spectral availability, spectral band count, and noise-floor availability.

The measurement-quality block does not change the gain decision. It explains how trustworthy the current analysis window is for FOH review and real-console validation.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 69 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 158 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE33.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE34.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE35.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE36.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE37.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE38.md`: passed.

Failures

Measurement readiness is heuristic. It does not yet validate whether analyzer values are physically plausible or time-aligned with the peak/LUFS windows.

The missing-noise-floor warning is informational; it does not automatically block all gain-up decisions because some test paths and real captures may not include inactive windows.

Critique

This improves pre-console readiness. During offline and real WING capture tests, an engineer can now separate "method chose the wrong action" from "the analyzer window was incomplete".

The method still needs real capture calibration for analyzer quality thresholds.

Next Research Step

Use real or capture-like WING analyzer streams to calibrate measurement-quality thresholds and decide which readiness states should block gain-up instead of only warning.
