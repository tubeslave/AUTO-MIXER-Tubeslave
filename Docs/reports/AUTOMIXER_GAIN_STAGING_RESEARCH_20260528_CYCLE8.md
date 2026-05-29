Gain Staging Research Report

Hypothesis

The method should not wait for actual clipping before protecting headroom. If recent peaks are both rising and densely clustered near the max peak limit, a small preemptive trim-down is safer than reacting after a digital clip event.

Changes

Added predictive clipping risk evidence to `SignalStats.calculate_input_trim_recommendation(...)`.

The method now computes:

- near-limit peak density;
- recent-vs-early peak trend;
- predicted peak estimate;
- bounded predictive trim-down.

When peak density and rising trend indicate likely clipping, the method uses `predictive_clip_risk` and applies a small capped cut. Live mode still caps the step to `live_max_cut_db`.

An isolated near-limit spike below the max peak limit is held as `isolated_spike_hold` instead of being treated as a predictive clip risk. Quiet spiky sources still go through the boost guard and remain blocked as `isolated_peak_boost_block`.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q` -> 23 passed.
- `python3 -m json.tool config/default_config.json >/dev/null` -> passed.
- `python3 -m py_compile backend/lufs_gain_staging.py` -> passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q` -> 157 passed, 45 existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q` -> 11 passed, 8 existing `datetime.utcnow()` warnings.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json` -> passed.

Failures

No focused regression found. The predictor is still based on short-window peak distribution, not a calibrated time-series model from real console meter history.

Critique

This is a safer FOH behavior because it trims before clipping while avoiding overreaction to one spike. The weak point is tuning: density/trend thresholds may need live dataset calibration for singers, drums, and playback tracks.

Next Research Step

Add rolling trend memory across several analysis windows: decay confidence when source behavior changes, but strengthen predictive trim-down when near-clip density repeats over time.
