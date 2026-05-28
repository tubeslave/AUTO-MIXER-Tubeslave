Gain Staging Research Report

Hypothesis

Not every hot input event means the same thing. A robust live gain-staging method should distinguish digital clipping, local preamp saturation risk, isolated spikes, and real vocal shouting or screaming before choosing trim-down, hold, or operator review.

Changes

Added overload classification to the input-trim recommendation report.

The method now classifies:

- `digital_clip` when true peak reaches the digital clip threshold;
- `preamp_saturation_risk` when a local input is already in the saturation zone and peak headroom is unsafe;
- `vocal_scream_or_shout` for hot vocal/speech windows with high peak and high integrated loudness;
- `isolated_spike` when peak spread suggests one event is dominating headroom.

Digital clipping forces a margin-based trim-down reason, `digital_clip_headroom`. Preamp saturation risk forces trim-down toward cleaner gain structure with `preamp_saturation_trim_down`. Vocal scream classification does not block trim-down; it makes the decision explainable.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q` -> 20 passed.
- `python3 -m json.tool config/default_config.json >/dev/null` -> passed.
- `python3 -m py_compile backend/lufs_gain_staging.py` -> passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q` -> 157 passed, 45 existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q` -> 11 passed, 8 existing `datetime.utcnow()` warnings.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE2.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE3.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE4.md` -> passed.

Failures

No focused regression found. The scream detector is threshold-based and does not yet use spectral harshness, sustained duration, or repeated phrase context.

Critique

The method is more explainable and safer around overload. The remaining weakness is classification depth: a clipped wireless pack, a vocalist scream, and analog preamp drive can overlap in real life. The current logic is conservative but not yet source-diagnostic enough.

Next Research Step

Add predictive clipping risk from percentile peak slope and recent peak density, so the method can trim down before clipping instead of only reacting after a hot event.
