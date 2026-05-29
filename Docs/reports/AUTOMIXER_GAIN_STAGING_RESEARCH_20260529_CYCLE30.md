Gain Staging Research Report

Hypothesis

Percussion gain staging should evaluate repeated hits, not just one aggregate attack/body split. A stable kick or snare pattern should show consistent body level and attack/body gap across hits; inconsistent hits should reduce confidence and stay clearly in engineer review.

Changes

Added repeated-hit segmentation inside the transient-window profile:

- detects contiguous attack runs after body windows;
- stores per-hit attack sample count and body sample count;
- stores per-hit attack p95, body median, and attack/body gap;
- computes hit-to-hit body standard deviation;
- computes hit-to-hit attack/body gap standard deviation;
- reports repeated-hit evaluability, body stability, gap stability, and instability.

Added confidence factor:

- `transient_hit_instability`: -0.12 when repeated percussion hits have unstable body level or attack/body gap.

Added default tuning:

- `transient_window_min_repeated_hits`: 3
- `transient_window_hit_body_std_max_db`: 3.0
- `transient_window_hit_gap_std_max_db`: 4.0

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 58 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 158 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.

Failures

Hit segmentation is still threshold/run based. It is not yet a true onset detector and can miss rolls, flams, ghost notes, or dense drum patterns.

The per-hit body window is only the body region immediately preceding an attack. That is practical for current metering windows, but not enough for full drum-envelope modeling.

Critique

This improves the method because percussion stability is now evaluated across repeated events. A stable kick pattern and an inconsistent kick pattern no longer look identical to the gain-staging confidence model.

The method remains conservative: drum gain-up is still operator-review-first. The new signal mostly improves explainability and confidence scoring rather than making automatic drum boosts easier.

Next Research Step

Research body-level target selection for percussion: use repeated-hit body medians as the safer gain reference while keeping attack peaks as headroom constraints.
