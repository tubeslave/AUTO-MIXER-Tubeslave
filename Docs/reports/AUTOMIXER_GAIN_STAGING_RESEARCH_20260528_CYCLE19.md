Gain Staging Research Report

Hypothesis

Time-aware confidence memory should decay gradually, not only expire at one hard boundary. Vocal confidence should decay faster than playback confidence because vocalist position, mic handling, and feedback context change more easily than a consistent playback source.

Changes

- Added source-family-specific confidence decay controls to `LocalInputGainProfile`.
- Decision confidence now reports memory decay family, half-life, memory strength, and minimum memory strength.
- Repeated borderline gain-up evidence now requires both freshness and enough remaining memory strength.
- Lead vocal, backing vocal, and speech use a shorter confidence half-life.
- Playback uses a longer confidence half-life, so stable playback evidence survives a longer gap than vocal evidence.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 42 passed.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE18.md`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.

Failures

- Decay constants are still heuristic and not learned from show data.
- Only confidence memory decays; channel profile and source signature memory still use simpler window logic.
- Playback decay is safer for consistent tracks but may be too permissive if the playback file changes without source-signature detection catching it.

Critique

The method became more FOH-aware because the same time gap is now treated differently for vocals and playback. This is closer to how live engineers reason about source stability. It is still incomplete because the half-life values are hand tuned and need replay validation against real soundcheck/live traces.

Next Research Step

Research coupling confidence decay with source-signature drift, so memory strength drops faster when the source signature slowly changes even if no hard identity-change threshold fires.
