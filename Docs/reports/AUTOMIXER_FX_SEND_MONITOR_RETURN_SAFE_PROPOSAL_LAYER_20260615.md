# FX Send Monitor/Return-safe Proposal Layer

Date: 2026-06-15
Status: implemented and validated

## Summary

FX / Sends now has a stricter proposal-only layer:

- `fx_send_proposal` adds monitor-safety evidence for FX/send analysis.
- Stale monitor context or high monitor risk keeps results `analysis_only`.
- Return-level safety is surfaced as explicit proposal evidence.
- Operator proposals include explicit `old_value` and `rollback_value` sources.
- Proposals remain non-queueable and not live-applicable.

## Safety Boundary

This does not contact WING, OSC, MIDI, or mixer clients. It does not create live
FX/send writes, does not import into a live queue, and does not grant live
authority.

## Validation

- `PYTHONPATH=backend python -m py_compile backend/fx_send_proposal.py backend/processing_proposal_adapters.py backend/processing_registry.py backend/project_goal_capability_matrix.py tests/test_fx_send_proposal.py tests/test_project_goal_capability_matrix.py`
  - PASS
- `PYTHONPATH=backend python -m pytest -q tests/test_fx_send_proposal.py tests/test_auto_fx.py tests/test_auto_reverb.py tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py`
  - `90 passed`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  - `3426 passed, 9 skipped`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
  - `status: passed`
  - `live_authority_granted: false`
  - `full_autonomous_ready: false`
- `/opt/homebrew/bin/git diff --check`
  - PASS
