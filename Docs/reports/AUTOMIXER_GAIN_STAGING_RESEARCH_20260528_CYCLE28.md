Gain Staging Research Report

Hypothesis

Drum and cymbal high-band brightness is often transient detail, not stable source identity. A gain-staging method should not let cymbal/snare/kick high-band spectral drift behave like vocal source continuity, and drum gain-up must remain operator-review-first.

Changes

Added transient high-band identity weighting:

- `source_signature_spectral_transient_high_band_identity_weight`: 0.25

For `transient_percussion`, `cymbal`, `drum_image`, and `drum_room` families, high/presence/air bands now use this lower identity weight in spectral drift distance. This separates transient brightness changes from stable input identity.

Added a regression proving that kick high-band drift is reported with low spectral identity weight while the channel still refuses automatic gain-up and stays in operator review.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 55 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.

Failures

The method still uses broad source-family labels. Kick, snare, overheads, and room mics should not share the same transient spectral identity model forever.

The transient model only changes identity weighting. It does not yet classify attack/decay windows or separate shell resonance from cymbal bleed.

Critique

This improves FOH safety because drum high-band brightness no longer has the same identity influence as vocal presence. The test also keeps the critical invariant: drum soundcheck gain-up remains operator-review-first, not automatic.

The method still needs a real transient analyzer. Current logic uses band names and family labels, not actual onset/decay segmentation.

Next Research Step

Research transient-window profiling: split attack, sustain, and decay windows for percussion so gain staging can distinguish shell/body level from cymbal/bleed spikes without touching artistic balance.
