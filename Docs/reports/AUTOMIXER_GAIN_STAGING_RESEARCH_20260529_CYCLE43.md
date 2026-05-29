Gain Staging Research Report

Cycle

43

Weakest Point

SafeGain ensemble context was available as detailed evidence rows, but the
top-level proposal reason still forced cockpit/read-only surfaces to parse those
rows before showing whether the context was active, stale, missing, or
operator-review-only.

Change

`operator_recommendation_bridge` now appends a compact ensemble summary to the
SafeGain proposal reason when `soundcheck_ensemble_context` is present. The
summary includes context state, canonical reason, time-coherence state,
source-count evidence, lag evidence, and operator-review-only policy.

Tests

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`:
  80 passed.
- `python3 -m py_compile backend/operator_recommendation_bridge.py backend/lufs_gain_staging.py`:
  passed.
- `python3 -m json.tool config/default_config.json`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`:
  158 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`:
  11 passed, existing `datetime.utcnow()` warnings.
- `python3 tools/memory/memory_lint.py`: passed.
- `git diff --check`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`:
  1242 passed, 1 skipped, 54 warnings.

Residual Weakness

The summary improves read-only operator visibility, but it does not calibrate
the ensemble time-coherence thresholds. The method remains research-grade until
real or replayed WING/Dante multichannel captures validate the lag and source
count rules.

Next Cycle Target

Either render the new summary explicitly in the cockpit task card, or calibrate
the stale/missing/coherent thresholds against saved multichannel capture data.

Live Safety

No live OSC/WING writes were executed. No real console was contacted. No fader,
gain, trim, EQ, compressor, routing, snapshot, reset, or mute command was run.
