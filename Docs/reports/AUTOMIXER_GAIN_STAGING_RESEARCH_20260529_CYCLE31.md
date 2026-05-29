Gain Staging Research Report

Hypothesis

For percussion, the body of repeated hits is a safer musical gain reference than the attack peak, but attack peaks must remain the hard headroom constraint. The method should expose this body-reference context without turning drum gain-up into automatic mixing.

Changes

Added percussion body-reference context inside the transient-window profile:

- median body peak across repeated hits;
- raw delta from body median to target peak;
- attack-headroom cap based on max true peak, max peak limit, and clip safety margin;
- headroom-limited delta;
- soundcheck/operator-review delta capped by `soundcheck_max_boost_db`;
- confidence flag requiring repeated-hit evaluability, stable body, stable attack/body gap, and positive attack headroom.

The body reference policy is explicitly `operator_review_only`. It does not enable automatic drum gain-up.

Added default tuning:

- `transient_window_body_reference_enabled`: true

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 58 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 158 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.

Failures

The body reference uses peak-domain body medians. It does not yet model short-term body RMS or shell resonance energy separately from spectral body bands.

The operator-review delta is useful evidence, but there is no UI/operator workflow consuming it yet.

Critique

This improves the method because it separates "where the body sits" from "where the attack peaks land". That is closer to FOH gain staging for drums: attack protects headroom, body informs whether the preamp/trim structure is under-driving the source.

The method remains conservative. Drum gain-up is still held for engineer review, so this does not become automixing.

Next Research Step

Research body-band spectral reference for percussion: use low/low-mid body stability as confidence evidence, while treating presence/air as transient or bleed-sensitive.
