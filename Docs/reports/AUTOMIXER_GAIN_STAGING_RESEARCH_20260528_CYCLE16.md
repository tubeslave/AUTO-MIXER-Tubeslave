Gain Staging Research Report

Hypothesis

Many separate binary holds make the gain-staging method harder to reason about. A FOH-safe method needs one explainable decision confidence score that combines source continuity, profile recovery, clipping risk, noise margin, channel personality, and stability evidence.

Changes

- Added decision-confidence controls to `LocalInputGainProfile`.
- Added `decision_confidence` evidence to the input-gain character report.
- Decision confidence now includes weighted factors for gain-up caution, headroom cuts, personality policy, distortion risk, source behavior, source identity, profile recovery, clipping risk, noise margin, repeated stability, boost guards, and anti-oscillation.
- `suggestion_confidence` now comes from the explainable score instead of a separate ad hoc confidence path.
- Soundcheck gain-up is held for operator review when the combined score is below the configured gain-up threshold.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 37 passed.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE12.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE13.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE14.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE15.md`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.

Failures

- The weights are heuristic and hand-tuned.
- Confidence is still window-local; it does not yet smooth or debounce scores near the threshold.
- The score explains why gain-up was blocked, but it does not yet learn from engineer acceptance or rejection.

Critique

The method became more explainable and safer because a marginal gain-up candidate can now be rejected even when no single binary guard fires. It is still not production-grade intelligence: the confidence model needs calibration against real show traces and should avoid jitter when a candidate hovers around the threshold.

Next Research Step

Research confidence hysteresis over time so a soundcheck boost is not allowed merely because one analysis window briefly crosses the score threshold.
