Gain Staging Research Report

Hypothesis

Spectral continuity should be treated as evidence only when the analyzer bands are stable enough to describe the source. Sparse bands, wildly moving bands, or stuck analyzer output should reduce inherited gain-up confidence instead of silently disappearing from the decision.

Changes

Added spectral stability metadata to each source fingerprint:

- per-band standard deviation;
- per-band span;
- signal LUFS/peak variability;
- combined signal variability.

Added spectral quality rejection rules:

- too few band samples;
- low band coverage;
- excessive band variance;
- excessive band span;
- stuck band output when the signal itself is changing.

Added a spectral quality penalty that reduces confidence memory for rejected spectral evidence:

- `source_signature_spectral_quality_penalty_per_rejected_band`: 0.15
- `source_signature_spectral_quality_max_penalty`: 0.55

The decision confidence report now exposes rejected band count, quality penalty tuning, and applied spectral quality penalty.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 53 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.

Failures

The stuck-output rule is still heuristic. A real analyzer could legitimately output stable spectral bands for a sustained source, so the rule only fires when LUFS or peak variability is present.

The variance thresholds are global. Cymbals, speech consonants, backing tracks, and bass instruments probably need different spectral instability limits.

No real WING/analyzer capture has verified these spectral quality fields yet.

Critique

The method is safer than Cycle 25 because bad spectral data now actively weakens gain-up memory. A sparse FFT frame, unstable bands, or stuck analyzer output can no longer be ignored while a borderline soundcheck boost inherits the previous confidence window.

This is still a research heuristic, not a final analyzer model. The next step should make spectral reliability source-family aware and connect it to a canonical band schema.

Next Research Step

Research source-family spectral quality profiles: vocals should reject unstable presence/air earlier, playback should tolerate broader mastered spectral motion, and transient percussion should require separate transient-aware spectral handling.
