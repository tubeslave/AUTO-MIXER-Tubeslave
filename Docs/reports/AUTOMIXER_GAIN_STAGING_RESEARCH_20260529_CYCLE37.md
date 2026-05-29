Gain Staging Research Report

Hypothesis

Smoothing should not assume that the previous recommendation changed the real console state. If the engineer dismissed or ignored a proposal, the next smoothing reference must reset to the current trim, not the ignored proposal.

Changes

Added applied-state-aware trim smoothing:

- reads previous proposal status fields such as `status`, `apply_status`, and `operator_status`;
- treats `dismissed`, `rejected`, `ignored`, `expired`, `apply_blocked`, and `apply_error` as explicitly unapplied;
- honors explicit `applied: false`;
- resets smoothing reference to current trim for explicitly unapplied previous proposals;
- reports `previous_apply_status`, `previous_applied`, `previous_explicitly_unapplied`, `reference_delta_db`, and `reference_source`.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 68 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 158 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE33.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE34.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE35.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE36.md`: passed.

Failures

Unknown previous status still assumes continuity for backward compatibility. This can be wrong if a caller omits status after the engineer ignored the proposal.

The method still depends on upstream proposal state being carried into `previous_recommendation`.

Critique

This improves FOH workflow fit. The method no longer treats an explicitly dismissed gain proposal as if it happened on the console, which prevents the smoothing model from drifting away from physical trim state.

The remaining weakness is state quality: the DSP method needs reliable operator/proposal lifecycle input to be fully trustworthy.

Next Research Step

Research confidence telemetry for final method readiness: produce a compact FOH-readable explanation of why a trim proposal is hold/up/down, including the dominant blockers and the exact evidence window state.
