Gain Staging Research Report

Hypothesis

Before testing on a real console, the method needs a compact FOH-readable decision summary. A WING validation run should show why a trim proposal is hold/up/down without forcing the engineer to inspect internal confidence, trend, transient, and guard structures.

Changes

Expanded `foh_summary`:

- exposed it at `input_gain_character.foh_summary` as well as inside `decision_confidence`;
- added `readiness`, `next_action`, and `safe_to_auto_apply`;
- added `risk_flags` for live gain-up block, peak-over-limit, overload, boost guards, vocal guard, trim smoothing, transient instability, and operator-review sources;
- added gain-structure snapshot: current trim, target trim, zones, distortion risk, and whether gain changes tone;
- added headroom snapshot: max true peak, p95 peak, target peak, max peak limit, predicted peak, near-clip density, and overload classification.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 68 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 158 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE33.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE34.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE35.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE36.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260529_CYCLE37.md`: passed.

Failures

The summary is still generated after the core decision rather than as a first-class decision object. A future refactor could make the decision and explanation a single typed result.

The summary is only as good as the upstream analyzer values and previous recommendation state.

Critique

This improves readiness for real console testing. The method now reports an operator-facing decision class and the core evidence needed to evaluate it quickly: auto-safe trim-down, soundcheck gain-up review, hold-for-review, or source check.

It remains conservative: the summary does not add apply authority, it only explains and classifies the existing gain-staging decision.

Next Research Step

Run offline capture-style scenarios with realistic analyzer band names and previous proposal statuses, then use any mismatches to tune the FOH summary and thresholds before connecting to WING.
