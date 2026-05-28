Gain Staging Research Report

Hypothesis

Not every hot peak means the same FOH action. Digital clipping, likely preamp saturation, a vocal scream, and an isolated spike should be classified separately so the method can explain why it cuts trim and avoid treating artistic loud moments as ordinary gain errors.

Changes

Added overload classification evidence to `SignalStats.calculate_input_trim_recommendation(...)`:

- `digital_clip`;
- `preamp_saturation_risk`;
- `vocal_scream_or_shout`;
- `isolated_spike`;
- `normal`.

Digital clipping forces a clip-margin trim-down decision. A local input already in the saturation-risk preamp zone with excessive peak forces trim-down toward the clean range. Vocal scream/shout is labeled but still uses normal peak-headroom trim-down logic rather than blocking protection.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q` -> 20 passed.
- `python3 -m json.tool config/default_config.json >/dev/null` -> passed.
- `python3 -m py_compile backend/lufs_gain_staging.py` -> passed.

Failures

The classifier is threshold-based, not source-trained. It cannot yet distinguish intentional vocal distortion from preamp overload without console-side saturation evidence.

Critique

This improves explainability and protects the most dangerous overload cases. The method still needs physical-console readback and longer signal history before it can claim reliable analog-overload diagnosis.

Next Research Step

Feed real WING readback and meter history into overload classification so preamp saturation decisions are based on both signal evidence and actual trim/preamp state.
