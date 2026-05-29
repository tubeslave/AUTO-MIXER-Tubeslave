Gain Staging Research Report

Hypothesis

Long-term channel profiling improves soundcheck gain-up safety, but a live FOH workflow needs lifecycle controls. A stale unstable profile should be resettable, lockable by the engineer, and decayed when old evidence is no longer representative.

Changes

- Added `profile_lifecycle` handling inside `SignalStats.calculate_input_trim_recommendation(...)`.
- Added lifecycle actions: `reset`, `lock`, `unlock`, and `decay`.
- `reset` clears stale channel-profile history before the current analysis window.
- `lock` freezes gain-up profile learning and blocks soundcheck gain-up for engineer review.
- Live headroom trim-down remains allowed even while the profile is locked.
- `decay` reduces the retained weight of previous profile windows before rolling averages are updated.
- Added profile lifecycle defaults to `config/default_config.json` and SafeGain status output.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 32 passed.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.

Failures

- Lifecycle requests currently ride on the previous recommendation payload; there is no dedicated operator command path in this cycle.
- Decay is a simple retained-window weight, not a full time-aware model.
- `noise_margin_min_db` remains conservative unless retained windows drop to zero, so recovery from old noisy-stage evidence can still be slow.

Critique

The method is safer and more FOH-friendly because it gives the engineer explicit authority over stale profile state without allowing live boosts. It is still incomplete as DSP intelligence: profile lifecycle is manual/request-driven, and decay is a heuristic rather than a calibrated time constant tied to source class, stage noise changes, or show phase.

Next Research Step

Research channel-profile recovery logic with separate short-term and long-term evidence lanes, so the method can detect when an unstable source has become stable without requiring a manual reset.
