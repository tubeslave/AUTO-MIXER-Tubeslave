Gain Staging Research Report

Hypothesis

A single analysis window with confidence barely above the gain-up threshold is not enough for live-sound safety. Borderline soundcheck boosts should require a repeated confidence window so the method does not react to score jitter.

Changes

- Added confidence hysteresis controls to `LocalInputGainProfile`.
- Decision confidence now tracks previous score, confidence windows, borderline state, required confidence windows, and score tolerance.
- Soundcheck gain-up is held when the confidence score is above the minimum threshold but still inside the borderline margin and lacks repeated stable confidence evidence.
- High-confidence gain-up can still pass without extra delay.
- Recovered vocal gain-up now requires both recovery evidence and repeated borderline-confidence evidence.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 39 passed.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE16.md`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.

Failures

- Confidence hysteresis is still count-based rather than time-based.
- The score tolerance is global and not source-family-specific.
- It does not yet distinguish a slow stable build-up from a brief but repeatable artifact.

Critique

The method is safer because marginal gain-up decisions now need temporal confirmation. It is also more predictable: high-confidence cases are not delayed, while borderline cases are held. The weak point is calibration; the hysteresis margin and tolerance need real trace validation per instrument family.

Next Research Step

Research time-aware confidence memory, so score history can decay by elapsed time and window duration instead of using only integer window counts.
