Gain Staging Research Report

Hypothesis

Automatic gain-staging decisions should not depend on the exact analyzer band labels. Real analyzers may report `63 Hz`, `250Hz`, `4 kHz`, or vendor-specific text instead of `sub`, `low_mid`, `presence`, and `air`.

Changes

Added spectral band canonicalization before storing analyzer data:

- numeric frequencies map into `sub`, `low`, `low_mid`, `mid`, `presence`, and `air`;
- semantic labels map into `sub`, `low`, `low_mid`, `body`, `fundamental`, `presence`, and `air`;
- multiple raw bands that map to the same canonical band are averaged per sample;
- transient-window and source-signature logic now consume canonical names.

Added tests proving:

- numeric analyzer bands produce canonical spectral fingerprints;
- kick repeated-hit body reference works from numeric analyzer bands;
- high-band transient detection still works after normalization.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 63 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 158 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE33.md`: passed.

Failures

Frequency boundaries are heuristic. The mapping from `600 Hz` through `2 kHz` into `mid` is not yet used by percussion body logic and may need instrument-specific tuning.

Vendor labels that do not include obvious text or frequency values still pass through as normalized raw labels.

Critique

This makes the method more production-ready because DSP decisions no longer require a perfectly named analyzer schema. It also reduces false instability when the analyzer reports numeric bands instead of semantic groups.

The method is still incomplete because canonicalization is not learned from the actual WING/analyzer capture format.

Next Research Step

Research vocal/speech adaptive gain-up gating using presence-band stability and scream/shout headroom history, so soft vocal gain-up is separated from unstable vocalist dynamics.
