Gain Staging Research Report

Hypothesis

Percussion gain staging needs a transient-window view, not only whole-window peak/LUFS and aggregate spectrum. Attack brightness can be cymbal/bleed/transient detail, while body/shell level is the safer input-gain reference.

Changes

Added active sample history to `SignalStats`, preserving peak, LUFS, and optional spectral bands per active sample.

Added transient-window profiling:

- splits active samples into attack and body windows by peak percentile;
- reports attack sample count and body sample count;
- measures attack peak p95, body peak median, and attack/body gap;
- measures high-band lift from body to attack;
- flags `attack_dominant` and `high_band_transient_dominant`.

Added transient attack high-band identity weighting:

- `source_signature_spectral_transient_attack_high_band_identity_weight`: 0.10

When a drum/cymbal source has high-band transient dominance, high/presence/air bands use the lower attack identity weight instead of the normal transient high-band identity weight.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 56 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 158 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.

Failures

The attack/body split is percentile-based. It does not yet use onset detection, envelope slope, or drum-specific timing.

The high-band lift model only looks at band names. It needs a canonical analyzer schema and real band calibration.

This still has no physical WING/analyzer proof.

Critique

This is a method improvement because it starts separating body level from attack brightness. It also preserves the FOH invariant that drum gain-up stays operator-review-first while making spectral identity less sensitive to transient high-band spikes.

It is not a full transient engine yet. The next step should model envelope attack, decay, and repeated hits instead of one percentile split.

Next Research Step

Research repeated-hit stability: detect whether percussion body level is stable across multiple hits, and keep gain-up in review if hit-to-hit body level or attack/body gap is inconsistent.
