Gain Staging Research Report

Hypothesis

Level-only source continuity is too weak for borderline soundcheck gain-up. Two different sources can share peak, LUFS, crest, and noise buckets while having a different tonal fingerprint. The method should preserve confidence memory for the same spectral shape, but weaken it when the tone shape changes.

Changes

Added optional active-signal spectral band collection to `SignalStats.update_sample(...)`.

Added a level-normalized spectral fingerprint to the source signature. It stores median band levels and relative band levels, so an overall gain shift does not look like a new source.

Added spectral source-signature drift penalty to decision confidence memory. The penalty uses the mean absolute relative-band distance across common bands, applies only when enough common bands exist, and does not become a hard source-identity change.

Added default tuning:

- `source_signature_spectral_drift_enabled`: true
- `source_signature_spectral_drift_min_common_bands`: 3
- `source_signature_spectral_drift_penalty_per_db`: 0.08
- `source_signature_spectral_drift_max_penalty`: 0.65

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 49 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json`: passed.

Failures

The method still depends on upstream analyzers providing meaningful band energies. Existing OSC/live metering paths do not yet prove that real console spectral bands are available or calibrated.

The spectral penalty is global, not instrument-specific. Vocal, guitar, drum, and playback sources probably need different band weights and tolerances.

The fingerprint currently uses median static bands only. It does not yet model formant movement, cymbal transients, plosive-heavy speech, or source changes that keep similar broad-band tone.

Critique

This is a real method improvement because it rejects an unsafe false continuity case without treating spectral drift as automix or artistic EQ. It remains conservative: large tone-shape drift only blocks inherited borderline gain-up, while live trim-down and hard clipping protection remain untouched.

It is not production-complete. A FOH engineer would still need analyzer calibration, per-source band semantics, and proof that spectral fingerprints are stable under microphone handling, bleed, and monitor spill.

Next Research Step

Research instrument-aware spectral drift weights: vocals should care more about presence and noise/air changes, playback should tolerate broader tonal changes, and drums should avoid interpreting transient-heavy cymbal energy as stable source identity.
