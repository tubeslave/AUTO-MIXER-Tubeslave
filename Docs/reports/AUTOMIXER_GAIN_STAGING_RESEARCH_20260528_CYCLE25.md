Gain Staging Research Report

Hypothesis

Spectral continuity should only influence gain-up confidence when the analyzer data is dense enough to be trusted. A single spectral sample inside a 20-sample gain-staging window can be noise, a transient, or a measurement artifact, so it should not prove or disprove source identity.

Changes

Added spectral quality metadata to the source fingerprint:

- per-band sample counts;
- per-band coverage ratios;
- active sample count.

Added a quality gate before spectral drift is computed:

- `source_signature_spectral_quality_gate_enabled`: true
- `source_signature_spectral_min_band_samples`: 3
- `source_signature_spectral_min_band_coverage`: 0.60

The decision confidence report now exposes all common spectral bands, rejected bands, rejection reasons, usable common bands, and the final spectral drift penalty.

Test Results

- `PYTHONPATH=backend python3 -m pytest tests/test_input_trim_autogain.py -q`: 51 passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_server.py -q`: 157 passed, existing `datetime.utcnow()` warnings.
- `PYTHONPATH=backend python3 -m pytest tests/test_lufs.py tests/test_gain_proof_dry_run_runner.py -q`: 11 passed, existing `datetime.utcnow()` warnings.
- `python3 -m json.tool config/default_config.json >/dev/null`: passed.
- `python3 -m py_compile backend/lufs_gain_staging.py`: passed.

Failures

Sparse spectral data is currently ignored rather than converted into lower decision confidence. That is conservative for avoiding false spectral penalties, but it may still permit a borderline gain-up when spectral data is unavailable and level-only continuity looks stable.

The quality gate is count/coverage based only. It does not yet detect noisy spectral bands, stuck analyzer output, or wildly unstable per-band values.

Critique

This makes the method safer and more explainable. Spectral continuity is now an evidence source with quality requirements, not a magic override. The report also tells an operator or test harness why a band did not count.

The method is still not analyzer-calibrated. A production FOH module should grade spectral quality using variance, band SNR, and analyzer schema identity, not just coverage.

Next Research Step

Research spectral stability scoring: per-band variance, stuck-band detection, and SNR-weighted spectral confidence, so unreliable bands lower gain-up confidence instead of silently disappearing.
