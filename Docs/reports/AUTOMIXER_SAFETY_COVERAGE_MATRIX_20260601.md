# Automixer Safety Coverage Matrix - 2026-06-01

## Summary

Added a read-only Safety Coverage Matrix for live-capable Automixer action
families. The goal is to separate "architecture exists" from "hardware proof
exists" before any live capability is widened.

## What Changed

- Added `backend/safety_coverage_matrix.py`.
- Exposed the matrix inside the existing `processing_pipeline_snapshot`.
- Added frontend adapter rows so the cockpit can display safety proof gaps in
  the existing module/status surfaces.
- Added tests for fail-closed status, proof-state validity, and frontend
  adapter mapping.

## Current Safety Interpretation

- `fader_supervised` is the only live-allowed row, and only as a narrow
  supervised path through `apply_operator_proposal`.
- `gain_supervised` remains dry-run ready, not hardware-proven.
- EQ, dynamics, gate, pan/phase/delay, FX, routing, snapshots, and scenes stay
  proposal-only, quarantined, or blocked.

## Why This Matters

This prevents the project from treating a working architecture as concert
readiness. Each action family now has explicit evidence fields:

- unit tests;
- offline harness;
- fake mixer proof;
- real WING proof;
- pre-read;
- apply readback;
- rollback readback;
- emergency-stop proof.

## Validation

Focused backend tests passed:

```bash
PYTHONPATH=backend python -m pytest tests/test_processing_registry.py tests/test_operator_product_state.py -q
```

Result:

```text
65 passed
```

## Remaining Work

- Calibrate telemetry alert thresholds on real/replay sessions.
- Complete a separate supervised WING gain proof.
- Add explicit connection-loss watchdog evidence before broad live validation.
- Keep all non-fader/gain actions blocked from live apply.
