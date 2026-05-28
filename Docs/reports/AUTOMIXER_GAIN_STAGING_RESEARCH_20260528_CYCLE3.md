Gain Staging Research Report

Hypothesis

A live-safe gain-staging method must avoid recommendation flip-flop. If repeated analyses alternate between small boost and small cut near the target window, the system can create operator mistrust, audible trim churn, and downstream compressor instability.

Changes

Added recommendation hysteresis to `SignalStats.calculate_input_trim_recommendation(...)`.

The method now accepts the previous recommendation and blocks small opposite-direction moves with `anti_oscillation_hold`. The hold is reported in `input_gain_character.hysteresis`.

Urgent peak/headroom cuts bypass hysteresis. A live trim-down caused by a peak above `max_peak_limit` is still allowed even if the previous recommendation was a boost.

`SafeGainCalibrator` now passes each channel's previous suggestion back into the next analysis cycle.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q` -> 16 passed.
- `python3 -m json.tool config/default_config.json >/dev/null` -> passed.
- `python3 -m py_compile backend/lufs_gain_staging.py` -> passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q` -> 157 passed, 45 existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q` -> 11 passed, 8 existing `datetime.utcnow()` warnings.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE2.md` -> passed.

Failures

No focused regression found. The hysteresis state is still one-step memory only; it does not yet track repeated trend confidence or the duration of stable evidence.

Critique

This reduces oscillation risk without blocking emergency cuts, which is the right FOH bias. It is still conservative and may hold a valid small correction after a previous opposite proposal. That is acceptable for live safety, but soundcheck could benefit from a multi-window trend model before deciding whether the correction is real.

Next Research Step

Add multi-window confidence accumulation: require stable evidence across several analysis windows before allowing gain-up, while allowing immediate trim-down for clipping or headroom risk.
