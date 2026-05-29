Gain Staging Research Report

Hypothesis

Source-signature drift fields should not all carry the same confidence penalty. In FOH terms, a noise-margin change is especially important for vocals, while LUFS and peak-gap drift are more meaningful for playback consistency.

Changes

- Added source-family-specific drift weights to `LocalInputGainProfile`.
- Drift penalty now uses the weighted sum of changed signature fields instead of raw field count.
- Vocal noise-margin and crest drift can reduce confidence memory more aggressively.
- Playback LUFS and peak-gap drift now carry a stronger reported weight while still benefiting from slower playback confidence decay.
- Decision confidence now reports field weights and weighted drift sum.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 45 passed.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE20.md`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.

Failures

- Field weights are still hand-tuned constants.
- Drift magnitude is still bucket-level; a small bucket crossing and a large in-bucket move are not distinguished.
- We still do not use spectral fingerprints, mic metadata, or patch state as source continuity evidence.

Critique

The method became more musical and FOH-aware because it now treats different evidence changes according to their live-sound meaning. It is safer for vocals around noise-floor changes and more explicit for playback consistency changes. The remaining weakness is that source signature remains coarse; weighted buckets are an improvement, not a full identity model.

Next Research Step

Research continuous source-signature distance, so drift penalty scales with actual dB movement rather than only bucket changes.
