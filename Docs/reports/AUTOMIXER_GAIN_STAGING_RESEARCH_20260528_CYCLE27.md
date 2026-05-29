Gain Staging Research Report

Hypothesis

Spectral quality thresholds should depend on source family. The same band variance is riskier on a vocal mic than on playback, and transient high-band sources need a different tolerance than steady tonal sources.

Changes

Added source-family spectral quality profiles:

- vocal presence/air bands use stricter variance and span limits;
- playback uses wider variance and span limits;
- transient percussion, cymbal, drum image, and room high bands use wider transient-aware limits.

Quality decisions now report which profile was used for each band: `general`, `vocal_presence`, `playback`, or `transient_high_band`.

Added default tuning:

- `source_signature_spectral_vocal_presence_max_std_db`: 4.0
- `source_signature_spectral_vocal_presence_max_span_db`: 10.0
- `source_signature_spectral_playback_max_band_std_db`: 9.0
- `source_signature_spectral_playback_max_band_span_db`: 20.0
- `source_signature_spectral_transient_high_band_max_std_db`: 10.0
- `source_signature_spectral_transient_high_band_max_span_db`: 24.0

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 54 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.

Failures

The transient profile is configured but still not proven against realistic drum/cymbal spectral sequences.

Playback tolerance is source-family wide. It does not distinguish full mastered tracks from click tracks, stems, or sparse playback channels.

The thresholds remain hand-tuned. They need calibration against real show captures.

Critique

The method improved because it stopped treating spectral analyzer quality as one universal rule. A vocal presence band can now be rejected earlier, while playback can preserve confidence under broader mastered spectral motion.

This is closer to FOH reasoning, but still incomplete. Real production behavior needs canonical analyzer band definitions and live capture-based threshold tuning.

Next Research Step

Research transient-aware spectral handling for drums and cymbals: separate short transient brightness from stable source identity, and ensure drum channels remain operator-review-first for gain-up.
