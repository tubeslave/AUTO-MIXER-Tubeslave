# WING MVP Proof Plan Required Summary Field Gate

## Summary

The no-WING `next_wing_proof_plan` schema gate now fails closed when the proof-plan summary omits any field declared by `required_supervised_proof_plan_summary_fields`.

## Changed

- `tools/no_wing_release_check.py` derives the required proof-plan summary fields from the shared WING handoff contract.
- The `next_wing_proof_plan_schema` detail now exposes `required_supervised_proof_plan_summary_fields` and `missing_required_supervised_proof_plan_summary_fields`.
- Missing required fields emit `next_wing_proof_plan_required_summary_fields_missing`.
- `tests/test_no_wing_release_check.py` covers a stale summary missing an accepted-candidate artifact-root field.

## Safety Boundary

This is an offline schema gate. It does not contact WING, query the console, grant live authority, or authorize any fader/gain movement. Missing required fields keep the proof path fail-closed.

## Validation

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_next_wing_proof_plan_schema tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_missing_required_summary_field` passed with 2 tests.

## Remaining Blocker

The WING MVP remains blocked on a real WING read-only preflight capture and operator-reviewed accepted candidate before any supervised fader/gain proof run.
