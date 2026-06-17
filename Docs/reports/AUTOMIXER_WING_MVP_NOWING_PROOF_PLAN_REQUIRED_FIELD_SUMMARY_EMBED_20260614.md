# WING MVP no-WING Proof Plan Required Field Summary Embed

## Summary

The aggregated no-WING `next_wing_proof_plan` summary now exposes the required and missing proof-plan summary field lists produced by `next_wing_proof_plan_schema`.

## Changed

- `tools/no_wing_release_check.py` propagates `required_supervised_proof_plan_summary_fields` into `next_wing_proof_plan`.
- `tools/no_wing_release_check.py` propagates `missing_required_supervised_proof_plan_summary_fields` into `next_wing_proof_plan`.
- `tests/test_no_wing_release_check.py` verifies the generated `summary.json` contains both fields.

## Safety Boundary

This is an offline summary propagation change. It does not contact WING, query the console, grant live authority, or authorize any fader/gain movement.

## Validation

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan` passed with 1 test.

## Remaining Blocker

The WING MVP remains blocked on a real WING read-only preflight capture and operator-reviewed accepted candidate before any supervised fader/gain proof run.
