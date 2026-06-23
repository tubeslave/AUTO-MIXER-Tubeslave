# Cross-channel Unmasking Arrangement Priority Gate

Date: 2026-06-15
Status: implemented and validated

## Summary

The proposal-only cross-channel unmasking layer now uses arrangement context:

- `channel_arrangement_context` can supply priority hints when explicit
  `channel_priorities` are absent.
- Explicit channel priorities remain authoritative when supplied.
- `arrangement_context_cache_freshness` is recorded in result metadata and
  per-action safety evidence.
- Provided stale arrangement context blocks linked mirror-EQ proposals with
  `cross_adaptive_eq_arrangement_context_cache_not_fresh`.
- Missing, unknown, non-processable, or low-confidence provided arrangement
  roles block output before any linked EQ proposal is emitted.

## Safety Boundary

The method remains transport-free and proposal-only. It does not contact WING,
create live writes, import proposals into a live queue, or grant live authority.

## Validation

- `PYTHONPATH=backend python -m py_compile backend/cross_adaptive_eq.py backend/source_identity_downstream_gate.py backend/frequency_separation_proposal.py backend/deesser_vocal_sibilance.py backend/processing_proposal_adapters.py backend/processing_registry.py backend/project_goal_capability_matrix.py tests/test_cross_adaptive_eq.py`
  - PASS
- `PYTHONPATH=backend python -m pytest -q tests/test_cross_adaptive_eq.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py`
  - `71 passed`
- `PYTHONPATH=backend python -m pytest -q tests/test_cross_adaptive_eq.py tests/test_source_identity_downstream_gate.py tests/test_frequency_separation_proposal.py tests/test_deesser_vocal_sibilance.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py`
  - `96 passed`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  - `3421 passed, 9 skipped`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
  - `status: passed`
  - `live_authority_granted: false`
  - `full_autonomous_ready: false`
- `PYTHONPATH=backend python tools/memory/memory_lint.py`
  - PASS
- `/opt/homebrew/bin/git diff --check`
  - PASS
