Gain Staging Research Report

Hypothesis

Soundcheck mode should converge faster than live mode when the input is far below target. A vocal that needs roughly +15..+21 dB should not be handled by the same small +3 dB cap and +1 dB smoothing used for cautious gain-up. The method still must avoid a single huge jump and must flag source/setup problems when the requested boost exceeds the clean input range.

Changes

Added a soundcheck convergence lane for large gain gaps:

- large-gap threshold defaults to 12 dB;
- vocal/speech convergence cap defaults to 9 dB;
- playback convergence cap defaults to 6 dB;
- general convergence cap defaults to 12 dB;
- convergence gain-up smoothing step defaults to 3 dB instead of the normal 1 dB;
- large requested boosts flag `source_setup_check_required` when they exceed the setup-check threshold or clean-range remaining headroom;
- `foh_summary` now exposes `soundcheck_convergence`, `large_soundcheck_gain_gap`, and `source_setup_check_required`.

Normal quiet-source behavior is preserved for ordinary gaps below the large-gap threshold.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 71 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 158 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json`: passed.

Failures

The first implementation over-flagged a normal +10 dB soft vocal as a source-setup problem because it exceeded clean-range remaining headroom. That was too conservative for soundcheck and was corrected so clean-range setup warnings apply only to large-gap cases.

Large convergence moves can still be held by confidence scoring if the proposed target enters coloration/saturation territory. This is intentional, but it means a profile must accurately describe the preamp clean range before real-console testing.

Critique

The method is now more FOH-usable for soundcheck: it can move faster toward the right input structure without changing live policy or touching faders. It still does not blindly chase +21 dB; it proposes staged movement and tells the engineer when the source chain should be checked.

The remaining weakness is that convergence step sizing is still window-based and heuristic. It does not yet estimate how quickly the engineer can safely audition each step in a real room.

Next Research Step

Test the convergence lane against real or recorded WING soundcheck captures: soft speech, quiet vocal mic, playback, compressed guitar, and noisy source. Calibrate the 12 dB large-gap threshold, 3 dB convergence step, and source-family caps from real console behavior.
