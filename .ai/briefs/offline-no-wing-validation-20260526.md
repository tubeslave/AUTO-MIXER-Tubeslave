# Offline No-WING Validation - 2026-05-26

## Task

Implement a repeatable Automixer validation path that does not require a real
WING Rack and cannot send live mixer writes.

## Scope

- Offline fixtures.
- Product Layer/operator analysis proposal generation.
- Fail-closed safety review.
- Simulated apply/readback/rollback.
- Emergency-stop block proof.
- Artifacts and report.

## Non-Goals

- No real mixer connection.
- No OSC/MIDI transport.
- No autonomous apply.
- No EQ, routing, snapshot, FX, training, or rule writes.

## Verification

- `PYTHONPATH=backend python -m pytest tests/test_offline_validation_harness.py -q`
- `python3 tools/offline_operator_validation_harness.py --artifact-root artifacts/offline_validation/current_no_wing`
