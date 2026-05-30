Gain Staging Research Report

Cycle

44

Weakest Point

SafeGain ensemble context could be explicitly disabled by the local input
preamp profile, but the report reason was selected after source-count checks.
In the calibrator path this could label a disabled feature as
`insufficient_loud_ensemble_sources`, which is misleading for cockpit evidence
and follow-up debugging.

Change

`SignalStats.calculate_input_trim_recommendation(...)` now reports
`soundcheck_ensemble_context.reason = "disabled"` before evaluating missing
source counts, vocal family, time coherence, reference loudness, or gain-gap
rules. The change is report/evidence-only: it does not alter live OSC/WING
writes, gain limits, boost caps, or supervised apply behavior.

Tests

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py::test_safe_gain_calibrator_reports_disabled_ensemble_context_before_source_counts -q`:
  failed before the fix with `insufficient_loud_ensemble_sources`, then passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`:
  81 passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `python3 -m json.tool config/default_config.json`: passed.
- `git diff --check`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`:
  160 passed.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`:
  1256 passed, 1 skipped, 1 existing third-party deprecation warning.

Residual Weakness

This cycle improves disabled-state evidence, not ensemble threshold
calibration. The time-coherence and source-count thresholds still need replayed
or real WING/Dante multichannel captures before the method can be treated as
hardware-proven.

Next Cycle Target

Use saved multichannel capture or replay artifacts to validate the ensemble
time-coherence lag and source-count rules, or render the existing ensemble
reason fields explicitly in the cockpit once the current frontend worktree is
cleanly staged.

Live Safety

No live OSC/WING writes were executed. No real console was contacted. No fader,
gain, trim, EQ, compressor, routing, snapshot, reset, or mute command was run.
