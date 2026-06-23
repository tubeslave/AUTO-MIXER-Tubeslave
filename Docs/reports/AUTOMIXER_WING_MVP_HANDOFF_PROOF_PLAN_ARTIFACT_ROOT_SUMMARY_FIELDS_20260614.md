# WING MVP Handoff Proof Plan Artifact Root Summary Fields

## Summary

The WING read-only handoff packet now requires accepted-candidate artifact-root proof-plan diagnostics in `required_supervised_proof_plan_summary_fields`. This makes the handoff contract match the downstream proof-plan/no-WING artifact-root gate.

## Changed

- `backend/wing_readonly_preflight_contract.py` adds accepted-candidate artifact-root fields to the shared required proof-plan summary field tuple.
- `tests/test_wing_readonly_preflight_handoff_packet.py` and `tests/test_no_wing_release_check.py` assert the handoff/no-WING required field list includes artifact-root diagnostics.
- `tests/test_wing_mvp_rails.py` and `tests/test_project_director_agent.py` assert the current command and selected action mention the root fields.
- Current WING blocker docs now describe artifact-root diagnostics as part of the handoff-required proof-plan summary field contract.

## Safety Boundary

This is an offline contract/reporting change. It does not contact WING, grant live authority, query the console, or authorize any fader/gain movement. Missing artifact-root fields keep the path fail-closed before supervised proof planning.

## Validation

- `PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py` passed with 295 tests and 8 skips.
- `PYTHONPATH=backend python -m pytest -q tests/test_wing_mvp_rails.py tests/test_project_director_agent.py` passed with 11 tests.
- `PYTHONPATH=backend python -m py_compile backend/wing_readonly_preflight_contract.py tools/build_wing_readonly_preflight_handoff_packet.py tools/no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py` passed.

## Remaining Blocker

The WING MVP remains blocked on a real WING read-only preflight capture and operator-reviewed accepted candidate before any supervised fader/gain proof run.
