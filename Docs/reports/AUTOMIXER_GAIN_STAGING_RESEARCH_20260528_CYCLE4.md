Gain Staging Research Report

Hypothesis

Soundcheck gain-up should require repeated stable evidence. A single quiet analysis window can be a singer holding back, an incomplete line check, room noise, bleed, or a temporary performance dip. Trim-down for peak risk should remain immediate.

Changes

Added multi-window gain-up stability state to `SignalStats.calculate_input_trim_recommendation(...)`.

The first soundcheck gain-up candidate is now held as `gain_up_stability_wait`. The report preserves the candidate direction, candidate delta, and evidence window count in `input_gain_character.stability`.

If the next analysis window repeats the same gain-up candidate within tolerance, the method allows the clean-range boost proposal. Live mode remains trim-down-only, and peak/headroom cuts remain immediate.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q` -> 17 passed.
- `python3 -m json.tool config/default_config.json >/dev/null` -> passed.
- `python3 -m py_compile backend/lufs_gain_staging.py` -> passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q` -> 157 passed, 45 existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q` -> 11 passed, 8 existing `datetime.utcnow()` warnings.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE2.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE3.md` -> passed.

Failures

No focused regression found. The stability model stores only a compact evidence count, not the full signal history or slope.

Critique

This is a useful live-safety bias: the method can cut immediately but must earn gain-up. The tradeoff is slower setup when a source is genuinely too quiet. For FOH this is acceptable because gain-up is the risky direction for noise, feedback, and preamp character.

Next Research Step

Add screaming/overload classification: distinguish a vocalist's real loud moment from analog overload, digital clipping, or isolated spike so the method chooses between immediate trim-down, hold, and operator source check.
