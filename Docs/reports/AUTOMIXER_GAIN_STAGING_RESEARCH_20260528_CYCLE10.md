Gain Staging Research Report

Hypothesis

Preset personality is not enough. A real live channel should build its own conservative profile over soundcheck windows, so the method can recognize that a specific mic, DI, or source has been unstable even if the current window looks safe.

Changes

Added compact per-channel profile learning to `SignalStats.calculate_input_trim_recommendation(...)`.

The method now rolls these channel-specific metrics through the previous recommendation payload:

- average crest factor;
- average LUFS variance;
- average near-clip peak density;
- minimum observed noise margin;
- unstable behavior ratio;
- profile readiness and gain-up block flag.

If soundcheck gain-up would otherwise be allowed but the channel profile shows too much instability, high loudness variance, or poor noise margin, the method holds with `channel_profile_stability_wait`.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q` -> 26 passed.
- `python3 -m json.tool config/default_config.json >/dev/null` -> passed.
- `python3 -m py_compile backend/lufs_gain_staging.py` -> passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q` -> 157 passed, 45 existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q` -> 11 passed, 8 existing `datetime.utcnow()` warnings.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE8.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE9.md` -> passed.

Failures

No focused regression found. The profile is still stored in the rolling recommendation payload, not a durable persisted per-channel database.

Critique

This is a stronger FOH method because a single good window cannot erase a risky channel history. It remains conservative and may delay gain-up for a channel that has improved. That is an acceptable bias for live input gain, but the next step should separate soundcheck learning from show-time operation more explicitly.

Next Research Step

Add soundcheck-vs-live profile phase behavior: soundcheck may learn and cautiously update channel profiles, while live should mostly freeze learned gain-up thresholds and only update risk/downward protection evidence.
