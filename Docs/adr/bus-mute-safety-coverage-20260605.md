# ADR: bus-mute-safety-coverage-20260605

## Context

The autonomy promotion readiness matrix includes promotion targets for
bus/group/DCA/matrix control and mute control. Before this decision, both
targets lacked safety coverage rows, so the readiness report had to emit a
generic `safety_coverage_row_missing` blocker.

## Options considered

1. Add fail-closed safety coverage rows and connect readiness to them.
2. Keep missing-row blockers until physical WING proof exists.
3. Promote proposal-only bus or mute behavior to supervised candidates.

## Decision

Use option 1. Add explicit safety rows for bus/group/matrix and mute, but keep
them quarantined and not live-allowed.

## Why this won

Explicit safety contracts make future proof work testable without expanding
automation authority. The system can now name the exact missing evidence for
these action families while preserving the current fader/gain live boundary.

## Rejected alternatives

Keeping missing-row blockers is less actionable. Promoting bus or mute actions
would be unsafe because physical readback, rollback, emergency-stop, and
headroom/intent proof are missing.

## Implementation plan

1. Add `bus_group_matrix` and `mute_supervised` rows to the safety coverage
   matrix.
2. Reference those rows from the corresponding promotion targets.
3. Update tests to prove the rows are fail-closed and no new live authority is
   granted.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_promotion_readiness.py tests/test_processing_registry.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk: operators may interpret a safety row as readiness. Rollback is removing
the two row references and safety rows; the readiness matrix will return to
generic missing-row blockers.
