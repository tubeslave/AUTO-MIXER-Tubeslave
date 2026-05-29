# Task proposal-only-correction-contracts-20260526: Safe proposal contracts for non-fader/gain corrections

## Problem

AUTO-MIXER already contains logic for EQ, compressor, gate, pan, phase, FX,
routing, and other correction domains, but the current proven WING live executor
is intentionally limited to supervised `fader` and `gain`. The cockpit needs a
clear Product Layer representation for the other correction domains without
accidentally making them queueable or executable.

## Constraints

- Preserve the current WING safety boundary.
- Do not add direct OSC, routing, EQ, compressor, FX, pan, phase, snapshot, or
  autonomous live writes.
- Keep `apply_operator_proposal` as the only apply path.
- Keep real live apply limited to supervised `fader` and `gain`.
- Avoid new production dependencies.

## Definition of Done

- Main pipeline preview exposes proposal-only contracts for non-fader/gain
  correction domains.
- Non-fader/gain proposals are marked non-applyable in the operator proposal
  queue.
- Each proposal-only contract documents target, old/new value, safety evidence,
  readback, approval, and rollback requirements.
- Focused backend tests pass.

## Test Command

`PYTHONPATH=backend python -m pytest tests/test_processing_registry.py tests/test_operator_product_state.py -q`
