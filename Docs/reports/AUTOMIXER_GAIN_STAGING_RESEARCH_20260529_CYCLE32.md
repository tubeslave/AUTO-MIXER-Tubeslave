Gain Staging Research Report

Hypothesis

Percussion body stability should use low/low-mid spectral evidence, not only peak-domain body level. High bands can be transient brightness or bleed, while low/low-mid bands better represent shell/body energy for gain staging.

Changes

Added body-band spectral reference inside transient-window hit profiles:

- extracts low/body/fundamental bands from each pre-attack body window;
- stores per-hit body-band median;
- computes hit-to-hit body-band standard deviation;
- reports body-band reference availability;
- reports stable and unstable body-band state.

Added confidence factor:

- `transient_body_band_instability`: -0.10 when repeated percussion body-band reference is unstable.

Added default tuning:

- `transient_window_body_band_reference_enabled`: true
- `transient_window_body_band_std_max_db`: 3.0

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 59 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 158 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.

Failures

The body-band selector still relies on band-name tokens. It needs a canonical analyzer band schema.

The low/low-mid body reference is tested with synthetic bands, not real kick/snare/overhead captures.

Critique

This improves the method because percussion body confidence now has a frequency-domain check separate from peak body level. A stable attack/body pattern with unstable low-band body energy is no longer treated as equally reliable.

The method remains conservative and FOH-friendly: it does not auto-boost drums; it strengthens the review evidence and confidence score.

Next Research Step

Research source-specific percussion body profiles: kick should prioritize low/fundamental bands, snare should prioritize low-mid/body bands, and overhead/room should not use the same body-band logic.
