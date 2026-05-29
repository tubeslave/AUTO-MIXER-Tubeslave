Gain Staging Research Report

Hypothesis

Percussion body reference must be source-specific. Kick, snare, tom, cymbal, overhead, and room channels do not expose the same gain-staging evidence in the same analyzer bands.

Changes

Added source-specific transient body-band profiles:

- `kick_low_fundamental`: prefers sub/low/fundamental bands and only falls back to low-mid/body.
- `snare_low_mid_body`: prefers low-mid/body bands so kick bleed in the low band does not destabilize snare reference.
- `tom_body_fundamental`: prefers body/fundamental/low-mid with low fallback.
- `cymbal_no_body_reference`, `drum_image_no_body_reference`, and `drum_room_no_body_reference`: disable per-hit body reference for high-band or image channels.

Added exact-ish body-band matching so `low` no longer automatically consumes `low_mid`.

Added report evidence:

- selected body-band profile;
- primary and secondary tokens;
- primary/secondary match counts per hit;
- whether body-band reference is applicable;
- non-applicable body-reference policy for cymbal/image/room channels.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 61 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 158 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json`: passed.

Failures

The profile selector still depends on preset names and analyzer band labels. If the incoming analyzer schema uses numeric bands instead of semantic labels, the profile can become unavailable.

The body-band profiles are synthetic FOH heuristics, not calibrated from real kick/snare/overhead captures yet.

Critique

The method improved because percussion body evidence is no longer one-size-fits-all. A snare channel can stay stable when low-band kick bleed moves, while overhead and room channels no longer get a misleading per-hit body-reference confidence.

The method remains conservative: no drum, cymbal, overhead, or room channel receives automatic gain-up from this evidence; it only improves operator-review context and confidence scoring.

Next Research Step

Research analyzer-schema normalization: map numeric or vendor-specific spectrum bands into canonical `sub`, `low`, `low_mid`, `body`, `presence`, and `air` groups before profile matching.
