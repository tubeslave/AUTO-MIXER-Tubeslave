Gain Staging Research Report

Hypothesis

Behavior labels alone are too coarse for safe recovery. A mic swap, performer change, or playback-file change can remain `stable_dynamic` while the gain structure risk changes. Soundcheck gain-up should wait when the source signature changes materially between windows.

Changes

- Added a coarse source signature to trend memory.
- Signature fields include channel family, source behavior, crest bucket, integrated LUFS bucket, peak-to-LUFS gap bucket, and noise-margin bucket.
- Added a source-identity change detector with configurable minimum changed fields.
- Soundcheck gain-up now holds for operator review when the source signature changes even if the behavior label did not flip.
- Added source signature config and SafeGain status fields.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 36 passed.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.
- `git diff --check -- backend/lufs_gain_staging.py tests/test_input_trim_autogain.py config/default_config.json Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE12.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE13.md Docs/reports/AUTOMIXER_GAIN_STAGING_RESEARCH_20260528_CYCLE14.md`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.

Failures

- The signature is a bucketed heuristic, not true source identification.
- It can hold gain-up after a legitimate performer dynamic change.
- It does not yet use spectral evidence, channel naming, patch metadata, or physical preamp state.

Critique

The method is safer because recovery and gain-up now require continuity of the source, not only stability of the latest window. It is still conservative and may produce false holds. That is acceptable for soundcheck gain-up, but it should become more discriminating before production use.

Next Research Step

Research confidence scoring that combines recovery, source continuity, clipping risk, noise margin, and personality policy into one explainable decision score instead of many separate binary holds.
