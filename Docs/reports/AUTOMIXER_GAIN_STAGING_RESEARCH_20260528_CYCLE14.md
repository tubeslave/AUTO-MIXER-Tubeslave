Gain Staging Research Report

Hypothesis

Channel-profile recovery should not use one global rule for every source. Vocals need slower recovery because dynamics and feedback risk are less predictable, while playback can recover faster when variance is low. Sources that already require operator review should not gain a recovery override.

Changes

- Added instrument-aware recovery policy using the existing `ChannelPersonalityProfile`.
- Lead vocal, backing vocal, and speech recovery now require at least 4 stable windows with stricter LUFS and peak variance limits.
- Playback recovery can clear stale long-term instability after 2 stable windows when the signal is consistent.
- Sources with `boost_policy == "operator_review"` cannot use recovery override for gain-up.
- Added recovery policy evidence to the channel profile report: family, allowed flag, thresholds, required windows, and recovery status.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 35 passed.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE12.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE13.md`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.

Failures

- Recovery policy is family-based, not learned from real channel behavior.
- Playback fast recovery assumes stable track-like material; it does not yet detect bad playback gain staging or distorted files.
- Vocal policy is conservative but still does not model feedback risk, mic handling, or performer distance.

Critique

The method became more FOH-aware because it now treats vocal recovery, playback recovery, and operator-review sources differently. It is still heuristic. The policy improves explainability and safety, but the thresholds are manually chosen and need validation against replayed show data before they should be trusted as production-grade.

Next Research Step

Research explicit source identity and source-change detection, so recovery cannot mistake a mic swap, performer change, or playback file change for a stable continuation of the same channel.
