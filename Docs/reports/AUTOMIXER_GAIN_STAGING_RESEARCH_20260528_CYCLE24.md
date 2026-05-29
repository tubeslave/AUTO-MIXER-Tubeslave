Gain Staging Research Report

Hypothesis

Spectral drift should not use one global tolerance for every source. A vocal presence/air shift is more suspicious for inherited gain-up confidence than a broad playback tone shift, while drum and cymbal high-band movement should be treated cautiously because transients can move spectral energy without proving stable input identity.

Changes

Added source-family spectral drift weights:

- Vocal and speech presence/air/high bands are weighted up.
- Playback spectral movement is weighted down.
- Drum, cymbal, drum image, and room high-band movement is weighted down to avoid over-reading transient brightness as source identity.

Changed spectral distance from plain mean absolute relative-band distance to weighted distance per common band. The decision confidence report now exposes per-band spectral weights, weighted distance, unweighted distance, and the resulting memory penalty.

Added default tuning:

- `source_signature_spectral_vocal_presence_weight`: 1.35
- `source_signature_spectral_playback_weight`: 0.55
- `source_signature_spectral_drum_high_band_weight`: 0.60

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 50 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.

Failures

Playback tolerance is still coarse. It lowers all spectral bands equally, which is reasonable for mastered tracks but may be too permissive for click tracks, stems, or playback channels that can change from one song to another.

The drum/cymbal high-band rule is implemented but not yet backed by a dedicated test with realistic transient band movement.

The spectral band names are conventional string tokens. The method still needs a canonical analyzer band schema before this can be called production-grade.

Critique

The method improved because it now respects source family. It rejects the same same-level spectral drift for a vocal while allowing playback to inherit confidence under a weighted tolerance. That is closer to FOH practice: vocal gain changes carry higher feedback and preamp-character risk, while playback is typically more controlled and less microphone-dependent.

The weak part is calibration. These weights are heuristic and should be trained against real show captures, especially vocal mic handling, stage bleed, and playback playlist changes.

Next Research Step

Research spectral availability and quality scoring. The method should distinguish calibrated analyzer bands from sparse, noisy, or inconsistent spectral data before using spectral continuity to influence gain-up confidence.
