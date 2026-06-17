# Pan Stereo Mono/Phase-safe Proposal Layer

Date: 2026-06-15
Status: implemented and validated

## Summary

Pan / Stereo Placement now has a stricter proposal-only layer:

- `pan_spatial_proposal` adds mono/phase evidence fields for pan analysis.
- Non-center pan moves are blocked unless mono/phase evidence is fresh and safe.
- Unsafe phase correlation or mono-sum loss keeps the result `analysis_only`.
- Operator proposals include explicit `old_value` and `rollback_value` sources.
- Proposals remain non-queueable and not live-applicable.

## Safety Boundary

This does not contact WING, OSC, MIDI, or mixer clients. It does not create a
live pan write, does not import into a live queue, and does not grant live
authority.

## Validation

- `PYTHONPATH=backend python -m py_compile backend/pan_spatial_proposal.py backend/processing_proposal_adapters.py backend/processing_registry.py backend/project_goal_capability_matrix.py tests/test_pan_spatial_proposal.py tests/test_project_goal_capability_matrix.py`
  - PASS
- `PYTHONPATH=backend python -m pytest -q tests/test_pan_spatial_proposal.py tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py`
  - `75 passed`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  - `3425 passed, 9 skipped`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
  - `status: passed`
  - `live_authority_granted: false`
  - `full_autonomous_ready: false`
- `/opt/homebrew/bin/git diff --check`
  - PASS
