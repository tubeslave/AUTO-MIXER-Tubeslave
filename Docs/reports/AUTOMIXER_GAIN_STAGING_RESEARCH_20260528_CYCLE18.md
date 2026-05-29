Gain Staging Research Report

Hypothesis

Confidence hysteresis should be time-aware. A borderline gain-up score from an old analysis window should not confirm a current soundcheck boost, even if the score value and candidate direction match.

Changes

- Added time-aware confidence memory controls to `LocalInputGainProfile`.
- `calculate_input_trim_recommendation(...)` now accepts optional `analysis_timestamp_sec` and `analysis_window_sec` evidence.
- Decision confidence now reports timestamp, analysis-window duration, elapsed time from previous confidence evidence, max allowed gap, and stale/expired state.
- Repeated borderline gain-up evidence only increments `confidence_windows` when the previous confidence window is fresh.
- Old or timing-less previous confidence evidence cannot confirm a borderline soundcheck boost.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 40 passed.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE16.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE17.md`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.

Failures

- When no explicit `analysis_window_sec` is supplied, duration is estimated from sample count and a default sample interval.
- Time memory still uses a hard expiry threshold rather than exponential decay.
- It does not yet account for source-specific acceptable gaps, such as playback tracks versus lead vocal checks.

Critique

The method is safer because stale confidence evidence can no longer unlock a borderline gain-up. It is also more explainable because the report shows why previous confidence was accepted or expired. The remaining weakness is that expiry is binary; real live workflow would benefit from gradual decay and source-family-specific timing.

Next Research Step

Research confidence decay curves per source family, so old confidence evidence loses weight gradually instead of switching from valid to expired at one hard boundary.
