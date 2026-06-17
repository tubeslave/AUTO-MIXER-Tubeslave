# Corrective EQ Protected-band Proposal Layer

Date: 2026-06-15
Status: implemented and validated

## Summary

EQ now has a safer proposal-only corrective layer:

- `corrective_eq_proposal` creates operator-facing EQ band proposals without
  live writes.
- Band id and band-frequency match are required before proposal emission.
- Feedback-notch and manual EQ bands are preserved and block proposal emission.
- Boosts are bounded by true-peak headroom and feedback-risk evidence.
- Fresh current EQ state, fresh phase context, source-identity trust, old-value
  readback source, and rollback source are explicit evidence.
- Proposals remain non-queueable and not live-applicable.

## Safety Boundary

This does not contact WING, OSC, MIDI, or mixer clients. It does not create live
EQ writes, does not import into a live queue, and does not grant live authority.

## Validation

- `PYTHONPATH=backend python -m py_compile backend/corrective_eq_proposal.py backend/processing_proposal_adapters.py backend/processing_registry.py backend/project_goal_capability_matrix.py tests/test_corrective_eq_proposal.py tests/test_project_goal_capability_matrix.py`
  - PASS
- `PYTHONPATH=backend python -m pytest -q tests/test_corrective_eq_proposal.py tests/test_auto_eq_fail_closed.py tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py`
  - `73 passed`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  - `3433 passed, 9 skipped`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
  - `status: passed`
  - `live_authority_granted: false`
  - `full_autonomous_ready: false`
- `PYTHONPATH=backend python tools/memory/memory_lint.py`
  - PASS
- `/opt/homebrew/bin/git diff --check`
  - PASS
