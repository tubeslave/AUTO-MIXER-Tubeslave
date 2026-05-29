Gain Staging Research Report

Hypothesis

Even when a gain-up decision is correct, repeated trim proposals should move like a slow engineering recommendation, not like an instantaneous level chaser. Urgent headroom cuts must bypass that smoothing.

Changes

Added trim suggestion smoothing:

- tracks previous smoothed delta from prior recommendation state;
- limits repeated gain-up proposal growth by `trim_smoothing_gain_up_step_db`;
- limits non-urgent gain-down proposal movement by `trim_smoothing_gain_down_step_db`;
- bypasses smoothing for urgent headroom cuts when true peak is over the configured max peak limit;
- reports smoothing state under `input_gain_character.trim_smoothing`.

Added default tuning:

- `trim_smoothing_enabled`: true
- `trim_smoothing_gain_up_step_db`: 1.0
- `trim_smoothing_gain_down_step_db`: 1.5

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 67 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 158 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE33.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE34.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE35.md`: passed.

Failures

The smoothing is step-based per analysis window, not yet time-normalized against actual analysis window length.

It uses previous recommendation state as its memory source. It does not yet know whether the previous proposal was actually applied by the operator.

Critique

This improves live safety because gain-up proposals now have slew-like behavior across repeated windows, reducing abrupt input gain jumps. Headroom protection remains fast: an over-limit peak still receives the full cut allowed by the safety policy.

The method remains conservative and does not perform automixing. It only shapes input trim proposal speed.

Next Research Step

Research applied-state-aware smoothing: distinguish previous proposal from confirmed applied trim so the method does not smooth against a recommendation the engineer ignored.
