Gain Staging Research Report

Hypothesis

Gain-up should not be trusted when the source behavior class changes between analysis windows. A channel that moves from low-noise-margin, sparse, transient, or unstable behavior into one apparently stable window should wait for consistent behavior before raising input gain.

Changes

Added compact trend memory to the input-trim recommendation payload.

The method now carries:

- recent source behavior history;
- previous and current source behavior;
- behavior flip flag;
- repeated predictive-clip window count;
- current near-clip density, peak trend, and noise margin.

If soundcheck gain-up would be allowed but the source behavior changed since the previous window, the method now holds with `behavior_change_stability_wait` and asks for another stable window.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q` -> 24 passed.
- `python3 -m json.tool config/default_config.json >/dev/null` -> passed.
- `python3 -m py_compile backend/lufs_gain_staging.py` -> passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q` -> 157 passed, 45 existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q` -> 11 passed, 8 existing `datetime.utcnow()` warnings.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE8.md` -> passed.

Failures

No focused regression found. Trend memory is still compact and piggybacks on the previous recommendation payload; it is not yet a dedicated persistent channel profiler.

Critique

This makes gain-up more FOH-safe because a single stable-looking window after a noisy or unstable window cannot immediately raise trim. It still needs a stronger long-term profile store to learn a specific channel's normal behavior over a whole show.

Next Research Step

Add per-channel long-term personality learning: accumulate safe ranges for peak density, crest factor, noise margin, and trim movement across soundcheck passes, then use that profile to adapt thresholds per channel.
