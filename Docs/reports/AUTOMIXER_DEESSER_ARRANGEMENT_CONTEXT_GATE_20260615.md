# De-esser Arrangement Context Gate

Date: 2026-06-15
Status: implemented and validated

## Summary

The proposal-only vocal de-esser now uses arrangement context:

- `arrangement_role` is recorded in recommendation evidence.
- Lead vocals keep the baseline de-esser profile.
- Tucked/backing/double/harmony vocals use a less aggressive profile.
- Provided stale arrangement context adds
  `deesser_arrangement_context_cache_not_fresh` and demotes output to
  `analysis_only`.

## Safety Boundary

The method remains transport-free and proposal-only. It does not contact WING,
create live writes, import proposals into a live queue, or grant live authority.

## Validation

- `PYTHONPATH=backend python -m pytest -q tests/test_deesser_vocal_sibilance.py tests/test_project_goal_capability_matrix.py`
  - `40 passed`
- `PYTHONPATH=backend python -m pytest -q tests/test_deesser_vocal_sibilance.py tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py tests/test_compressor_dynamics_proposal.py tests/test_pan_spatial_proposal.py tests/test_bus_group_balance_proposal.py`
  - `109 passed`
- `PYTHONPATH=backend python -m py_compile backend/deesser_vocal_sibilance.py backend/processing_proposal_adapters.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
  - PASS
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  - `3418 passed, 9 skipped`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
  - `status: passed`
  - `live_authority_granted: false`
  - `full_autonomous_ready: false`
- `PYTHONPATH=backend python tools/memory/memory_lint.py`
  - PASS
- `/opt/homebrew/bin/git diff --check`
  - PASS
