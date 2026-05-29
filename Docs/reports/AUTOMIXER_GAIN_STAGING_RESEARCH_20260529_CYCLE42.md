Gain Staging Research Report

Cycle

42

Weakest Point

Active ensemble-assisted SafeGain evidence could show the state as `active`
without carrying the canonical reason on the same row. That made the evidence
less self-explanatory for downstream read-only UI surfaces.

Change

`operator_recommendation_bridge` now includes `reason` on the `Ensemble context`
evidence row. A new regression covers an active ensemble-assisted gain-up and
checks that the evidence carries active state, reason, coherent time context,
source-count, lag, and `operator_review_only` policy facts.

Tests

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 79 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 158 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`: 1241 passed, 1 skipped, 54 warnings.
- `python3 -m py_compile backend/operator_recommendation_bridge.py backend/lufs_gain_staging.py`: passed.
- `python3 -m json.tool config/default_config.json`: passed.
- `python3 tools/memory/memory_lint.py`: passed.
- `git diff --check`: passed.

Residual Weakness

This improves operator-facing explanation but does not calibrate ensemble
time-coherence thresholds. Active ensemble context remains research-grade until
validated against real or replayed WING/Dante multichannel captures.

Next Cycle Target

Build a read-only cockpit summary for active/stale/missing ensemble context, or
calibrate the lag/sample thresholds against saved multichannel captures.

Live Safety

No live OSC/WING writes were executed. No real console was contacted.
