# Proposal: bus-mute-safety-coverage-20260605

## 1. Short thesis

Replace missing safety coverage for bus/group/matrix and mute promotion targets
with explicit fail-closed safety contracts.

## 2. What I understood

The project is moving toward autonomous console control, but broad surfaces must
first have machine-readable safety proof requirements. Current readiness rows
say the safety row is missing, which is less useful than naming the exact proof
gates that block promotion.

## 3. Proposed solution

Add two `SafetyCoverageRow` entries:

- `bus_group_matrix`
- `mute_supervised`

Both stay `quarantined`, `live_allowed=false`, and list missing fake-mixer,
real-WING, pre-read, apply-readback, rollback-readback, and emergency-stop
proof. Wire the matching promotion targets to these row IDs.

## 4. Likely files to touch

- `backend/safety_coverage_matrix.py`
- `backend/autonomy_promotion_readiness.py`
- `tests/test_processing_registry.py`
- `tests/test_autonomy_promotion_readiness.py`
- `Docs/adr/bus-mute-safety-coverage-20260605.md`

## 5. Alternatives considered

- Leave missing-row blockers as-is. Rejected because the next proof work cannot
  distinguish an unknown contract from a known fail-closed contract.
- Mark bus or mute proposal-only instead of quarantined. Rejected because these
  are direct console control families with high live-sound risk.

## 6. Risks

The rows could be mistaken for progress toward live apply. Tests should assert
that no live authority is added and only fader remains supervised-live-allowed.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_promotion_readiness.py tests/test_processing_registry.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q`
- `python3 tools/memory/memory_lint.py`

## 8. Where the other agent may disagree

Kimi may prefer keeping mute separate from broad bus/matrix safety. I agree the
proof paths differ, so the rows stay separate while landing in one small safety
coverage slice.
