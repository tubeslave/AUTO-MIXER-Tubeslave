Gain Staging Research Report

Hypothesis

Soundcheck and live should not update the same channel profile fields. Soundcheck can learn gain-up behavior cautiously, but live should freeze gain-up learning and only accumulate risk/downward-protection evidence. Otherwise a live scream, spike, or noisy moment can poison the learned soundcheck profile and make future gain-up decisions too erratic.

Changes

Added profile phase behavior to `SignalStats.calculate_input_trim_recommendation(...)`.

The channel profile now reports:

- `phase`;
- `learning_mode`;
- `gain_up_profile_frozen`;
- live risk window count;
- live overload event count;
- live peak-density maximum;
- current live risk flag.

In `soundcheck`, the profile continues learning crest, LUFS variance, near-clip density, noise margin, and unstable behavior ratio.

In `live`, gain-up profile fields are frozen by default. Live windows update only separate risk evidence, while trim-down protection remains immediate and capped by live policy.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q` -> 27 passed.
- `python3 -m json.tool config/default_config.json >/dev/null` -> passed.
- `python3 -m py_compile backend/lufs_gain_staging.py` -> passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q` -> 157 passed, 45 existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q` -> 11 passed, 8 existing `datetime.utcnow()` warnings.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE8.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE9.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE10.md` -> passed.

Failures

No focused regression found. The phase split is still carried in the recommendation payload and is not yet persisted as a dedicated per-channel profile store.

Critique

This makes the method more live-safe: live no longer retrains gain-up behavior just because the show had a hot or noisy window. The method still needs durable channel identity so profiles can survive across calibration passes and be reset intentionally by the engineer.

Next Research Step

Add explicit channel profile lifecycle controls: reset, lock, decay, and engineer override reasons, so learned gain-staging behavior can be managed predictably across soundcheck and live.
