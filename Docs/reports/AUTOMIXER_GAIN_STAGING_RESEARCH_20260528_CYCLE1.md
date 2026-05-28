Gain Staging Research Report

Hypothesis

Soundcheck boost should not be decided by peak error alone. A safe automatic input-gain raise needs enough source presence, moderate crest factor, and an integrated level far enough above the noise floor. Otherwise the method can raise preamp gain on sparse hits, drum transients, bleed, or room noise.

Changes

Added a soundcheck-only boost guard to `LocalInputGainProfile` and `SignalStats.calculate_input_trim_recommendation(...)`.

The guard blocks automatic boost when:

- signal presence is below `soundcheck_boost_min_presence`;
- crest factor is above `soundcheck_boost_max_crest_db`;
- integrated LUFS is below `soundcheck_boost_min_lufs`.

Live mode remains trim-down-only. Trim-down recommendations are unchanged. Stable soft-vocal soundcheck boost still works, but it remains proposal-only and cannot auto-apply.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q` -> 8 passed.
- `python3 -m json.tool config/default_config.json >/dev/null` -> passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_operator_analysis_does_not_boost_quiet_sources tests/test_server.py -q` -> 111 passed, 35 deprecation warnings from `datetime.utcnow()`.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json` -> passed.

Failures

No regression found in the focused tests. The method still uses coarse static thresholds; it does not yet learn per-channel personality or distinguish a real quiet source from quiet bleed with a dedicated noise-floor estimator.

Critique

This is safer than peak-only boost logic, but it is still a first-order heuristic. A fixed `16 dB` crest threshold can be too strict for some vocals and too loose for some percussion. Presence ratio depends on analysis-window length and gate threshold, so sparse-but-valid soundcheck material may need operator review instead of automatic boost. I would trust it as a conservative proposal generator, not as autonomous live gain-up logic.

Next Research Step

Add channel personality modeling: track rolling crest, peak spread, LUFS stability, and noise-floor estimate per preset so boost/cut confidence adapts to vocal, drum, guitar, DI, and playback behavior instead of relying only on global thresholds.
