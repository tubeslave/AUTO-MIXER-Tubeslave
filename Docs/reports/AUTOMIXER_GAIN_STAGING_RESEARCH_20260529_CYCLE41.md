Gain Staging Research Report

Hypothesis

Solo vocal soundcheck and ensemble soundcheck should not use the same gain-up caution. If only one microphone is active, a quiet vocal can be a singer-distance or speech-level artifact, so the method should stay conservative. If other instruments are already active and loud enough, and the vocal is far below that stage context, the input stage is more likely under-gained and can use a bolder soundcheck-only gain step.

Changes

Added ensemble-context awareness to the input gain staging method:

- `calculate_input_trim_recommendation` now accepts optional `ensemble_context`;
- `SafeGainCalibrator` builds ensemble context automatically from other active/loud channels before final recommendations;
- vocal/speech gain-up can become ensemble-assisted when other loud sources meet count and level thresholds;
- ensemble-assisted vocal gain-up uses a 6 dB cap/step by default instead of the solo +3 dB cap;
- the method reports `soundcheck_ensemble_context` in `input_gain_character` and `foh_summary`;
- confidence gets a small positive factor when ensemble context confirms the vocal is under-staged;
- faders and artistic balance remain untouched.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 74 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 158 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json`: passed.

Failures

The ensemble logic is currently level-context based. It does not yet know whether those other loud channels are actually part of the current soundcheck song, a line-check burst, or background playback.

The reference is built from other channel peak/LUFS percentiles. This is practical, but it still needs calibration from real WING captures.

Critique

This improves the method in the direction the FOH workflow needs: caution is highest during isolated mic checks, and bolder decisions become available only when the stage context proves the vocal input is under-staged. The method still avoids automixing because it does not compare fader balance or push vocal level for artistic prominence.

The remaining risk is false confidence if other channels are loud but unrelated to the current source check.

Next Research Step

Add time-coherence checks for ensemble context: require other loud sources and the vocal under-stage evidence to overlap in the same recent analysis window before applying the ensemble-assisted gain-up lane.
