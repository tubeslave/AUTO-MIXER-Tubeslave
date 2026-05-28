Gain Staging Research Report

Hypothesis

Global boost guards are safer than peak-only logic, but they still treat a lead vocal, kick drum, playback feed, guitar, and speech mic too similarly. Automatic input gain should adapt to source personality, measured loudness stability, and measured noise-floor margin before proposing a soundcheck boost.

Changes

Added preset-level channel personality profiles to the input-trim method. The method now reports source family, boost policy, adaptive crest/stability thresholds, and confidence bias.

Added inactive-window noise-floor tracking and `noise_floor_margin_db`. Soundcheck boost is blocked when a source is too close to measured noise floor.

Added operator-review boost policy for drums, cymbals, overheads, and room sources. These sources can still receive trim-down headroom protection, but automatic boost becomes an operator review item instead of a proposed apply.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q` -> 14 passed.
- `python3 -m json.tool config/default_config.json >/dev/null` -> passed.
- `python3 -m py_compile backend/lufs_gain_staging.py` -> passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q` -> 155 passed, 45 existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q` -> 11 passed, 8 existing `datetime.utcnow()` warnings.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json` -> passed.

Failures

No focused regression found. The personality table is still heuristic and static. It does not yet learn a specific singer, drummer, DI, or room across repeated soundchecks.

Critique

The method is safer and more FOH-friendly because it blocks gain-up on noisy, unstable, and percussion-like sources. It is still not a learned model. The noise floor estimate only uses inactive windows, so a continuously noisy but always-active channel may still need a stronger estimator.

Next Research Step

Add recommendation hysteresis and anti-oscillation state: repeated analyses should not alternate boost/hold/cut around the target window, especially for unstable vocalists and transient-heavy instruments.
