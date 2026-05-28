Gain Staging Research Report

Hypothesis

Gain-up is riskier than gain-down in live input gain staging. A soundcheck boost should not happen after one analysis window; the same gain-up evidence should repeat before the method proposes raising preamp/trim.

Changes

Added repeated-evidence gating for soundcheck gain-up:

- `gain_up_min_stable_windows`;
- `stable_window_tolerance_db`;
- `input_gain_character.stability`.

The method now holds first-window gain-up as `gain_up_stability_wait`. If the next recommendation has the same direction and similar candidate delta, the evidence counter advances and the boost can proceed through the existing boost guards. Live trim-down and urgent headroom cuts are not delayed.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q` -> 17 passed.
- `python3 -m json.tool config/default_config.json >/dev/null` -> passed.
- `python3 -m py_compile backend/lufs_gain_staging.py` -> passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q` -> 157 passed, 45 existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q` -> 11 passed, 8 existing `datetime.utcnow()` warnings.

Failures

No focused regression found. The method still stores only compact previous recommendation evidence, not a full rolling history.

Critique

This is a strong live-safety bias: gain-up becomes slower and more deliberate, while gain-down remains immediate for headroom risk. The tradeoff is that a genuinely quiet source may require one extra soundcheck window before boost is proposed. That is acceptable because input gain-up can change feedback margin, preamp tone, and downstream compressor behavior.

Next Research Step

Move from one-step repeated evidence to rolling trend memory: keep several windows of source behavior, candidate delta, noise margin, and peak distribution, then decay confidence when the channel flips behavior classes.
