Gain Staging Research Report

Hypothesis

A long-term unstable channel profile should not block soundcheck gain-up forever if the current source has become stable. The method needs a short-term recovery lane that can override stale long-term risk only after repeated stable evidence.

Changes

- Added channel-profile recovery controls to `LocalInputGainProfile`.
- Added `recovery_windows` tracking inside the channel profile report.
- Added strict recovery criteria: soundcheck only, not locked/frozen, stable source behavior, low LUFS variance, low peak variance, bounded crest factor, and sufficient noise margin.
- Long-term history still computes a raw block flag, but stable short-term recovery can suppress that block after the configured window count.
- Recovery is disabled during live risk-only mode and while the channel profile is locked.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 34 passed.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE12.md`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.

Failures

- Recovery windows are count-based, not time-based.
- The recovery lane still uses simple variance thresholds instead of instrument-specific dynamics.
- It does not yet distinguish a singer warming up from a different performer or mic swap.

Critique

The method improved because stale long-term instability no longer requires manual reset when stable evidence repeats. It remains conservative: one stable window is not enough, live mode cannot recover gain-up, and lock still wins. The weak point is calibration; the recovery thresholds are generic and may be too slow for controlled backing tracks or too permissive for volatile vocals.

Next Research Step

Research instrument-aware recovery thresholds, especially vocal versus backing-track behavior, so recovery speed follows source personality rather than one global window rule.
