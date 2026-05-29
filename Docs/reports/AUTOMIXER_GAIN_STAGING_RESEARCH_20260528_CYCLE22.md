Gain Staging Research Report

Hypothesis

Bucket-only source-signature drift misses meaningful in-bucket changes. Confidence memory should decay from continuous dB movement even when crest, LUFS, peak-gap, or noise-margin buckets do not change.

Changes

- Added raw continuous source-signature fields: crest, integrated LUFS, peak-to-LUFS gap, and noise-margin dB.
- Added continuous source-signature distance to decision confidence.
- Continuous drift uses the same source-family field weights as bucket drift.
- Confidence memory now applies both bucket-crossing penalty and continuous in-bucket drift penalty.
- Added report evidence for per-field continuous distance, normalization, weights, weighted units, and final continuous penalty.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 47 passed.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE21.md`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.

Failures

- Continuous distance is still limited to simple level-domain features.
- The distance normalizers are fixed at bucket widths rather than learned from source variance.
- Spectral shape, mic proximity, and room bleed are still not modeled.

Critique

The method is more sensitive and safer because confidence memory can now weaken before a bucket crossing. Small vocal noise-margin movement can keep a borderline boost alive, while a larger in-bucket movement can force another confidence window. This is closer to FOH behavior, but the source identity model is still incomplete without spectral or patch metadata.

Next Research Step

Research spectral source-continuity features, starting with low-cost band-energy fingerprints that can distinguish mic/source changes better than level-only signatures.
