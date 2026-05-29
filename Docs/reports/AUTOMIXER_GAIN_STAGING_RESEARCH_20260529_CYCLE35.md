Gain Staging Research Report

Hypothesis

Soft vocal gain-up should be separated from unstable vocalist dynamics. A quiet vocal window is not automatically safe if the presence band is moving sharply or the channel recently showed scream/shout behavior.

Changes

Added a vocal/speech gain-up guard:

- evaluates current `presence` and `air` band standard deviation/span;
- blocks soundcheck gain-up when available presence-band evidence is unstable;
- tracks `vocal_scream_cooldown_windows` in trend memory;
- blocks gain-up for a short cooldown after a scream/shout window;
- does not block trim-down or headroom protection.

Added default tuning:

- `vocal_gain_up_guard_enabled`: true
- `vocal_gain_up_presence_min_bands`: 1
- `vocal_gain_up_presence_max_std_db`: 4.0
- `vocal_gain_up_presence_max_span_db`: 10.0
- `vocal_gain_up_scream_cooldown_windows`: 2

Added report evidence under `vocal_gain_up_guard` and `trend_memory.vocal_gain_up_guard`.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 65 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 158 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE33.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE34.md`: passed.

Failures

The guard uses presence/air band stability only. It does not yet distinguish intentional vocal tone change from mic handling, proximity shift, or harsh shout behavior.

The scream cooldown is window-count based, not time-constant based.

Critique

The method is safer for live vocals: it avoids boosting a channel immediately after shout behavior and refuses gain-up when the presence band is unstable. It remains FOH-friendly because it still only touches input gain structure and does not chase vocal level with faders.

This is still conservative and may hold some valid gain-up proposals until a stable window appears.

Next Research Step

Research time-constant smoothing for gain suggestions: separate immediate headroom cuts from slower gain-up confidence, and expose attack/release-like behavior for trim proposals without creating pumping.
