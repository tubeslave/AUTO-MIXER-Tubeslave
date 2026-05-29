Gain Staging Research Report

Hypothesis

Source-signature drift should weaken confidence memory before the hard source-identity threshold fires. A single changed signature bucket may not prove a new source, but it should reduce trust in previous borderline gain-up evidence.

Changes

- Added source-signature drift penalty controls to `LocalInputGainProfile`.
- Decision confidence now reports time-memory strength before drift, drift fields, drift field count, drift penalty, and final memory strength.
- Confidence memory is reduced when signature buckets change but do not yet trigger the hard source-identity hold.
- Vocal borderline gain-up can be held by a single signature-bucket drift because vocal confidence already decays quickly.
- Playback can retain enough confidence for the same single-bucket drift when time evidence is fresh and source behavior remains stable.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 44 passed.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE19.md`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.

Failures

- Drift is still bucket-count based; it does not measure continuous spectral or envelope distance.
- The same per-field drift penalty applies to all signature fields.
- Playback drift remains permissive and depends on the coarse source-signature detector catching file changes.

Critique

The method is safer because confidence memory now degrades before a binary identity change. This better matches FOH reasoning: a channel can be "probably the same source" but still less trustworthy for gain-up. The weak point is resolution; bucket drift is explainable and testable, but it is still a rough proxy for real source continuity.

Next Research Step

Research weighted source-signature drift, where crest, LUFS, peak-gap, and noise-margin changes have different penalties per source family.
