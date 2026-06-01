Gain Staging Research Report

Cycle

45

Weakest Point

SafeGain ensemble context already reported whether source-count and
time-coherence gates passed, but it did not expose the signed margin for those
gates. That made replay calibration harder: a stale vocal ensemble with one
missing coherent source looked the same as a badly undersampled ensemble unless
the reader manually subtracted thresholds from counts and lag values.

Change

`SignalStats.calculate_input_trim_recommendation(...)` and
`SafeGainCalibrator._soundcheck_ensemble_context_for_channel(...)` now include
explicit evidence-only margins:

- `other_loud_source_margin`
- `time_coherent_source_margin`
- `reference_lag_margin_samples`

Positive values show spare evidence, zero means exactly on threshold, and
negative values show how far the ensemble context missed the gate. The change
does not alter gain decisions, thresholds, live OSC/WING writes, boost caps,
supervised apply behavior, or operator authority.

Tests

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py::test_soundcheck_ensemble_context_reports_source_and_lag_margins -q`: 1 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 83 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 161 passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `python3 -m json.tool config/default_config.json`: passed.
- `git diff --check`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`: 1349 passed, 1 skipped, 1 existing third-party deprecation warning.

Residual Weakness

The margin fields make ensemble calibration easier to inspect, but the
thresholds are still heuristic. The next real improvement needs replayed
multichannel captures or WING/Dante captures to tune source-count and
time-coherence limits against actual stage behavior.

Next Cycle Target

Use saved multichannel capture or replay artifacts to compare the new margins
against accepted/rejected vocal gain-up cases, then tune thresholds only if the
captures prove a safer rule.

Live Safety

No live OSC/WING writes were executed. No real console was contacted. No fader,
gain, trim, EQ, compressor, routing, snapshot, reset, or mute command was run.
