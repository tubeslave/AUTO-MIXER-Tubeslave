# Compressor Feedback-safe Proposal Layer

Date: 2026-06-15
Status: implemented and validated

## Summary

Compression now has a stricter proposal-only safety contract:

- `compressor_dynamics_proposal` blocks high-feedback-risk makeup proposals.
- Feedback safety is explicit recommendation evidence.
- Operator proposals include old-value and rollback-value readback sources.
- The proposal contract now requires feedback-risk and rollback/readback
  evidence.
- Compressor backlog status is now `proposal_only`.
- Proposals remain non-queueable and not live-applicable.

## Safety Boundary

This does not contact WING, OSC, MIDI, or mixer clients. It does not create live
compressor writes, does not import into a live queue, and does not grant live
authority.

## Validation

- `PYTHONPATH=backend python -m py_compile backend/compressor_dynamics_proposal.py backend/processing_proposal_adapters.py backend/processing_registry.py backend/project_goal_capability_matrix.py tests/test_compressor_dynamics_proposal.py tests/test_project_goal_capability_matrix.py`
  - PASS
- `PYTHONPATH=backend python -m pytest -q tests/test_compressor_dynamics_proposal.py tests/test_auto_compressor_fail_closed.py tests/test_compressor.py tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py`
  - `87 passed`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  - `3434 passed, 9 skipped`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
  - `status: passed`
  - `live_authority_granted: false`
  - `full_autonomous_ready: false`
- `PYTHONPATH=backend python tools/memory/memory_lint.py`
  - PASS
- `/opt/homebrew/bin/git diff --check`
  - PASS
